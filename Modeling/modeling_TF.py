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
from torch.utils.data import Dataset
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm, trange
import re
import time
import json


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from metrics import metrics
from modeling_utils_TF import LazyTextDataset


logger = logging.getLogger(__name__)
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']

######################################################################################
# Model integrating the representations of the top K terms with the highest TF score #
######################################################################################

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

        last_hidden_states = outputs[0]
        
        token_type_ids= token_type_ids.unsqueeze(-1)
        marker_masks=marker_masks.transpose(0,1)
        last_state = outputs[1] #cls
        # For each term in top-K
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

#==========================

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
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
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

def evaluate(args, model, tokenizer, prefix="", set_name='dev', global_step):
    eval_outputs_dirs = (args.output_dir,) #eval / dev /both
    eval_dataset_paths = (f'{args.data_dir}/doc2query_run/TF/run_dev.csv',) #eval path / dev path/both
    all_metrics = np.zeros(len(METRICS_MAP))
    print(eval_dataset_paths)
    for dataset_path, eval_output_dir in zip(eval_dataset_paths,eval_outputs_dirs):
        eval_dataset= LazyTextDataset(dataset_path, tokenizer,args.max_seq_length)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        
        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3],
                          'marker_masks': batch[4]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.msmarco_output:
            msmarco_file = open(
                f"{args.output_dir}/TF/checkpoint-{global_step}/msmarco_predictions_{set_name}.tsv", "w")
        query_docids_map = []
        with open(
            f"{args.data_dir}/doc2query_run/TF/query_doc_ids.top1000.doc2query.dev.small_full.txt") as ref_file:
          for _ in range(args.chunk*1745000):
            next(ref_file)
          for line in ref_file:
            query_docids_map.append(line.strip().split("\t"))

        start_time = time.time()
        results = []
        all_metrics = np.zeros(len(METRICS_MAP))
        example_idx = 0
        total_count = 0
        for log_probs,label_ids in zip(preds,out_label_ids):
            results.append((log_probs, label_ids))
            if total_count % 10000 == 0:
                logger.info("Read {} examples in {} secs".format(
                    total_count, int(time.time() - start_time)))

            if len(results) == args.num_eval_docs:

                log_probs, labels = zip(*results)
                log_probs = np.stack(log_probs).reshape(-1, 2)
                labels = np.stack(labels)

                scores = log_probs[:, 1]
                pred_docs = scores.argsort()[::-1]
                gt = set(list(np.where(labels > 0)[0]))

                all_metrics += metrics(
                    gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)

                if args.msmarco_output:
                    start_idx = example_idx * args.num_eval_docs
                    end_idx = (example_idx + 1) * args.num_eval_docs
                    query_ids, doc_ids = zip(*query_docids_map[start_idx:end_idx])
                    assert len(set(query_ids)) == 1, "!!!!Query ids must be all the same!!!!!"
                    query_id = query_ids[0]
                    rank = 1
                    for doc_idx in pred_docs:
                        doc_id = doc_ids[doc_idx]
                        # Skip fake docs, as they are only used to ensure that each query
                        # has 1000 docs.
                        if doc_id != "00000000":
                            msmarco_file.write(
                                "\t".join((query_id, doc_id, str(rank))) + "\n")
                            rank += 1

                example_idx += 1
                results = []

            total_count += 1

        if args.msmarco_output:
            msmarco_file.close()

        all_metrics /= example_idx

        logger.info("Eval {}:".format(set_name))
        logger.info("  ".join(METRICS_MAP))
        logger.info(all_metrics)
        



def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

     ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--msmarco_output", action='store_true',
                        help="Return msmarco output format file")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu=1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
    #                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    
    # Set seed
    set_seed(args)
    num_labels=2
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tokenizer = BertWordPieceTokenizer(f"{args.data_dir}/bert_based_uncased_vocab.txt", lowercase=True)
    tokenizer.enable_truncation(args.max_seq_length)
    tokenizer.enable_padding('right',max_length=args.max_seq_length)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    model.to(args.device)

    args.output_mode='classification'

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        dataset_path = f'{args.data_dir}/triples.unique.eq.train.small.csv'
        train_dataset=LazyTextDataset(dataset_path, tokenizer,args.max_seq_length)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = BertWordPieceTokenizer(f"{args.data_dir}/bert_based_uncased_vocab.txt", lowercase=True)
        tokenizer.enable_truncation(args.max_seq_length)
        tokenizer.enable_padding('right',max_length=args.max_seq_length)
        checkpoints = [args.output_dir] # can specifiy only one checkpoint checkpoints = [f'{args.data_dir}/checkpoint-{args.checkpoint}']
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = MarkersBertForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, tokenizer, prefix=prefix, set_name='eval', global_step)
            

    return results


if __name__ == "__main__":
    main()