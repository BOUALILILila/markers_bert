from __future__ import absolute_import, division, print_function
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig,BertForSequenceClassification, BertModel, BertPreTrainedModel
from tokenizers import BertWordPieceTokenizer

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, RandomSampler,  SequentialSampler, DataLoader, Dataset
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
import linecache
import csv
import re
import time
import json

# local module
import six
from six.moves import range
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import metrics


logger = logging.getLogger(__name__)
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']



def rolling_window(a, size):
          shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
          strides = a.strides + (a. strides[-1],)
          return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def encode(tokenizer,  query, 
                       doc, 
                       label, 
                       max_length, 
                       K,
                       codes,
                       pad_on_left= False, 
                       mask_padding_with_zero= True, 
                       pad_token=0, 
                       pad_token_segment_id=0):
                               
        inputs = tokenizer.encode(
                query,
                doc,
            )
        input_ids, token_type_ids = inputs.ids, inputs.type_ids

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        #transform to a list top K, K in args
        markers_indices=[]
        in_ids=np.array(input_ids)
        #K=2, 87.61%
        for i in range(K):
          marker_indices=np.array([0]*512)
          bool_indices=np.all(rolling_window(in_ids,4)==[1031,1041,codes[i],1033],axis=1)
          indices=np.mgrid[0:len(bool_indices)][bool_indices]
          for j in range(4):
            marker_indices[indices+j]=1
          markers_indices.append(torch.tensor(marker_indices))
        markers_indices=torch.stack(markers_indices)
        
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        return torch.tensor(input_ids), torch.tensor(attention_mask),torch.tensor(token_type_ids), torch.tensor(int(label)), markers_indices
        
class LazyTextDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=512,K=2):
        self._filename=filename
        with open(filename, "r") as f:
          for i, l in enumerate(f):
            pass
        self._total_data = i
        self.tokenizer=tokenizer
        self._max_len=max_len
        self.K=K
        self.codes={}
        for i in range(K):
          self.codes[i]=tokenizer.encode(f'e{i}').ids[2]

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx+2)
        csv_line = csv.reader([line], delimiter=',')
        row= next(csv_line)
        ###change indices of row
        return encode(self.tokenizer,row[4],row[5],row[3],self._max_len, self.K,self.codes)  
          
    def __len__(self):
        return self._total_data

class MarkersBertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = MarkersBertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, K=2):
        super(MarkersBertForSequenceClassification, self).__init__(config)

        self.max_size = (2*K+1)* config.hidden_size

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.max_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        marker_masks=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        logger.info("bert out")
        last_hidden_states = outputs[0]
        
        token_type_ids= token_type_ids.unsqueeze(-1)
        marker_masks=marker_masks.transpose(0,1)
        last_state = outputs[1] #cls
        for mask in marker_masks:
            mask = mask.unsqueeze(-1)
            markers_last_hidden_states= last_hidden_states*mask

            q_markers_last_hidden_states= markers_last_hidden_states*(1+(-1)*token_type_ids)
            d_markers_last_hidden_states= markers_last_hidden_states*token_type_ids

            q_markers_state= torch.sum(q_markers_last_hidden_states, axis=1)
            d_markers_state= torch.sum(d_markers_last_hidden_states, axis=1)

            last_state= torch.cat((last_state, q_markers_state, d_markers_state), dim=1)

        # dropout 
        last_state = self.dropout(last_state)

        logits = self.classifier(last_state)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs  # (loss), logits, (hidden_states), (attentions)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # gotta have the total train dataset size
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs

    # assuming max_steps always given and < total steps for 1 epoch
    #t_total = args.max_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num steps = %d", t_total)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataset, desc="Iteration", disable=args.local_rank not in [-1, 0])
        #epoch_iterator = tqdm(train_dataset.per_device_loader(args.device), desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3],
                      'marker_masks': batch[4]}
            outputs = model(**inputs)
            
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                #xm.optimizer_step(optimizer, barrier=True)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))
                    epoch_iterator.set_postfix(loss=loss_scalar, lr=learning_rate_scalar, step=global_step)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_dir", default=None, type=str, required=True,
                        help="The vocab file for the tokenizer.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--top_K", default=2, type=int,
                        help="The maximum number of markers to consider.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_eval_docs", default=1000, type=int,
                        help="number of docs per query in eval set.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--msmarco_output", action='store_true',
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.n_gpu = 1
    #args.device = xm.xla_device()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    # Set seed
    set_seed(args)
    num_labels=2
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tokenizer = BertWordPieceTokenizer(f"{args.vocab_dir}/vocab.txt", lowercase=True)
    tokenizer.enable_truncation(args.max_seq_length)
    tokenizer.enable_padding('right',max_length=args.max_seq_length)
    model = MarkersBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config,  K=args.top_K)
    #model.to(args.device)

    args.output_mode='classification'

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        dataset_path = f"{args.data_dir}/marked_TF_triples.unique.eq.train.small.csv"
        train_dataset=LazyTextDataset(dataset_path, tokenizer,args.max_seq_length)
        train_sampler = RandomSampler(train_dataset) 
        dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
        global_step, tr_loss = train(args, dataloader, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

if __name__ == "__main__":
    main()