"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import tensorflow as tf
import time
# local module
import six
from six.moves import range
MAX_LEN=512

#-----------------------

def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
      outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
      try:
        outputs = six.ensure_text(outputs, "utf-8")
      except UnicodeDecodeError:
        outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
      outputs = outputs.lower()

    return outputs

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
      if isinstance(text, str):
        return text
      elif isinstance(text, bytes):
        return six.ensure_text(text, "utf-8", "ignore")
      else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
      if isinstance(text, str):
        return six.ensure_text(text, "utf-8", "ignore")
      elif isinstance(text, six.text_type):
        return text
      else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
      raise ValueError("Not running on Python2 or Python 3?")

def write_to_tf_record(writer,
                       tokenizer, 
                       query, 
                       docs, 
                       labels, 
                       max_length, 
                       pad_on_left= False, 
                       mask_padding_with_zero= True, 
                       pad_token=0, 
                       pad_token_segment_id=0, 
                       ids_file= None, 
                       query_id= None, 
                       doc_ids= None):
                       
    query = convert_to_unicode(query)
    for i, (doc_text, label) in enumerate(zip(docs, labels)):
        doc_text = convert_to_unicode(doc_text)
        inputs = tokenizer.encode_plus(
                query,
                doc_text,
                add_special_tokens=True,
                max_length=max_length,
            )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

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
        
        input_ids_tf = tf.train.Feature(
          int64_list=tf.train.Int64List(value= input_ids))
        
        attention_mask_tf = tf.train.Feature(
          int64_list=tf.train.Int64List(value= attention_mask))

        token_type_tf = tf.train.Feature(
          int64_list=tf.train.Int64List(value= token_type_ids))

        labels_tf = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[label]))

        features = tf.train.Features(feature={
            'input_ids': input_ids_tf,
            'attention_mask': attention_mask_tf,
            'token_type_ids' : token_type_tf, 
            'label': labels_tf,
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

        if ids_file:
          ids_file.write('\t'.join([query_id, doc_ids[i]]) + '\n')
#--------------------------
