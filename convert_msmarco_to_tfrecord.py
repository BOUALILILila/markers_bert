""
import collections
import os
import re
import tensorflow as tf
import time
# local module
from transformers import BertTokenizer
import tokenization
from absl import app
from absl import flags



FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_folder", None,
    "Folder where the tfrecord files will be written.")

flags.DEFINE_string(
    "train_dataset_path",
    "./data/triples.train.small.tsv",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "dev_dataset_path",
    "./data/top1000.dev.tsv",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "eval_dataset_path",
    "./data/top1000.eval.tsv",
    "Path to the MSMARCO eval dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "dev_qrels_path",
    "./data/qrels.dev.tsv",
    "Path to the query_id relevant doc ids mapping.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_eval_docs", 1000,
    "The maximum number of docs per query for dev and eval sets.")


def convert_eval_dataset(set_name, tokenizer):
    print('Converting {} set to tfrecord...'.format(set_name))
    start_time = time.time()

    if set_name == 'dev':
      dataset_path = FLAGS.dev_dataset_path
      relevant_pairs = set()
      with open(FLAGS.dev_qrels_path) as f:
        for line in f:
          query_id, _, doc_id, _ = line.strip().split('\t')
          relevant_pairs.add('\t'.join([query_id, doc_id]))
    else:
      dataset_path = FLAGS.eval_dataset_path

    queries_docs = collections.defaultdict(list)  
    query_ids = {}
    with open(dataset_path, 'r') as f:
      for i, line in enumerate(f):
        query_id, doc_id, query, doc = line.strip().split('\t')
        label = 0
        if set_name == 'dev':
          if '\t'.join([query_id, doc_id]) in relevant_pairs:
            label = 1
        queries_docs[query].append((doc_id, doc, label))
        query_ids[query] = query_id

    # Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
    queries = list(queries_docs.keys())  # Need to copy keys before iterating.
    for query in queries:
      docs = queries_docs[query]
      docs += max(
          0, FLAGS.num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
      queries_docs[query] = docs

    assert len(
        set(len(docs) == FLAGS.num_eval_docs for docs in queries_docs.values())) == 1, (
            'Not all queries have {} docs'.format(FLAGS.num_eval_docs))

    writer = tf.io.TFRecordWriter(
        FLAGS.output_folder + '/dataset_' + set_name + '.tf')

    query_doc_ids_path = (
        FLAGS.output_folder + '/query_doc_ids_' + set_name + '.txt')
    with open(query_doc_ids_path, 'w') as ids_file:
      for i, (query, doc_ids_docs) in enumerate(queries_docs.items()):
        doc_ids, docs, labels = zip(*doc_ids_docs)
        query_id = query_ids[query]

        tokenization.write_to_tf_record(writer=writer,
                          tokenizer=tokenizer,
                          query=query, 
                          docs=docs, 
                          labels=labels,
                          max_length= FLAGS.max_seq_length,
                          ids_file=ids_file,
                          query_id=query_id,
                          doc_ids=doc_ids)

        if i % 100 == 0:
          print('Writing {} set, query {} of {}'.format(
              set_name, i, len(queries_docs)))
          time_passed = time.time() - start_time
          hours_remaining = (
              len(queries_docs) - i) * time_passed / (max(1.0, i) * 3600)
          print('Estimated hours remaining to write the {} set: {}'.format(
              set_name, hours_remaining))
    writer.close()


def convert_train_dataset(tokenizer):
    print('Converting to Train to tfrecord...')

    start_time = time.time()

    print('Counting number of examples...')
    num_lines = sum(1 for line in open(FLAGS.train_dataset_path, 'r'))
    print('{} examples found.'.format(num_lines))
    writer = tf.io.TFRecordWriter(
        FLAGS.output_folder + '/dataset_train.tf')

    with open(FLAGS.train_dataset_path, 'r') as f:
      for i, line in enumerate(f):
        if i % 1000 == 0:
          time_passed = int(time.time() - start_time)
          print('Processed training set, line {} of {} in {} sec'.format(
              i, num_lines, time_passed))
          hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
          print('Estimated hours remaining to write the training set: {}'.format(
              hours_remaining))

        query, positive_doc, negative_doc = line.rstrip().split('\t')

        tokenization.write_to_tf_record(writer=writer,
                          tokenizer=tokenizer,
                          query=query, 
                          docs=[positive_doc, negative_doc], 
                          labels=[1, 0],
                          max_length= FLAGS.max_seq_length)

    writer.close()


def main(argv):

    print('Loading Tokenizer...')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if not os.path.exists(FLAGS.output_folder):
      os.mkdir(FLAGS.output_folder)

    convert_train_dataset(tokenizer=tokenizer)
    convert_eval_dataset(set_name='dev', tokenizer=tokenizer)
    convert_eval_dataset(set_name='eval', tokenizer=tokenizer)
    print('Done!')  

if __name__ == '__main__':
    app.run(main)