from transformers import BertConfig, TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from modeling_utils import get_dataset
import argparse
import logging
import os


logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer):

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=args.adam_epsilon, clipnorm=args.max_grad_norm)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    history = model.fit(train_dataset, steps_per_epoch=args.max_steps)

    model.save_pretrained(args.output_dir)

    return history.history['loss'][0]

def evaluate(args, model, tokenizer, prefix="", set_name='dev'):
    eval_outputs_dirs = (args.output_dir,) #eval / dev /both
    eval_dataset_paths = (f'{args.data_dir}/dataset_{set_name}.tf')
    for dataset_path, eval_output_dir in zip(eval_dataset_paths,eval_outputs_dirs):
        eval_dataset= get_dataset(dataset_path, args.eval_batch_size, args.max_seq_length)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        
        msmarco_file = tf.io.gfile.GFile(
            args.output_dir + "/msmarco_predictions_" + set_name + ".tsv", "w")
        query_docids_map = []
        with tf.io.gfile.GFile(
            args.data_dir + "/query_doc_ids_" + set_name + ".txt") as ref_file:
            for line in ref_file:
                query_docids_map.append(line.strip().split("\t"))
        
        result= model.evaluate(eval_dataset, args.eval_batch_size)
        logger.info(" average loss= %s, accuaracy = %s", result[0], result[1])

        start_time = time.time()
        results = []
        all_metrics = np.zeros(len(METRICS_MAP))
        example_idx = 0
        total_count = 0
        for item in result:
            results.append((item["log_probs"], item["label_ids"]))
            if total_count % 10000 == 0:
                logger.info("Read {} examples in {} secs".format(
                total_count, int(time.time() - start_time)))

            if len(results) == FLAGS.num_eval_docs:

                log_probs, labels = zip(*results)
                log_probs = np.stack(log_probs).reshape(-1, 2)
                labels = np.stack(labels)

                scores = log_probs[:, 1]
                pred_docs = scores.argsort()[::-1]
                gt = set(list(np.where(labels > 0)[0]))

                all_metrics += metrics.metrics(
                    gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)

                if FLAGS.msmarco_output:
                    start_idx = example_idx * FLAGS.num_eval_docs
                    end_idx = (example_idx + 1) * FLAGS.num_eval_docs
                    query_ids, doc_ids = zip(*query_docids_map[start_idx:end_idx])
                    assert len(set(query_ids)) == 1, "Query ids must be all the same."
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

        if FLAGS.msmarco_output:
            msmarco_file.close()

        all_metrics /= example_idx

        logger.info("Eval {}:".format(set_name))
        logger.info("  ".join(METRICS_MAP))
        logger.info(all_metrics)

def main():
    METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Optional
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")    
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
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO )


    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_steps)
        dataset_path = f'{args.data_dir}/dataset_train.tf'
        train_dataset = get_dataset(dataset_path, args.per_gpu_train_batch_size, args.max_seq_length, args.do_train)
        tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" Done training !")
        logger.info(" average loss = %s", tr_loss)
        

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", args.eval_batch_size)


if __name__ == "__main__":
    main()