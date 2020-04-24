import argparse
import nltk
from nltk.stem.porter import PorterStemmer
import time, re
import tensorflow as tf
from transformers import BertTokenizer
from Data_processing import clean_text, write_to_tf_record
import spacy as sp
from spacy.tokens import Doc

def clean_text(text):
    #encoding
    try:
        t = text.encode("ISO 8859-1")
        enc_text = t.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        enc_text = text

    #line break
    text= enc_text.replace('\n',' ')
    
    #empty characters
    text = " ".join(text.strip().split())

    return text


def marker(query, doc, nlp, porter):
    d = nlp(clean_text(doc))
    q = nlp(clean_text(query))
    marked_p = []
    marked_q = []
    stem_to_id = dict()
    q_i = 0
    for token in q:
        marked_q.append(token.text)
        if not (token.is_punct or token.is_stop):
            stem = porter.stem(token.text.lower())
            marked_q.pop()
            if stem in stem_to_id :
                i = stem_to_id[stem]
                marked_q.append(f"[e{i}]{token.text}[\e{i}]")
            if stem not in stem_to_id :
                stem_to_id[stem] = q_i
                marked_q.append(f"[e{q_i}]{token.text}[\e{q_i}]")
                q_i +=1
            
    for i,term in enumerate(d):
        marked_p.append(term.text)
        if not (term.is_punct or term.is_stop):
            stem = porter.stem(term.text.lower())
            for q_stem in stem_to_id:
                if q_stem == stem:
                    q_i = stem_to_id[stem]
                    marked_p.pop()
                    marked_p.append(f"[e{q_i}]{term.text}[\e{q_i}]")
                    break  

    qu = Doc(nlp.vocab, words=marked_q, spaces= [token.whitespace_ for token in q])    
    doc = Doc(nlp.vocab, words=marked_p, spaces=[token.whitespace_ for token in d])
    return ''.join(token.text_with_ws for token in qu), ''.join(token.text_with_ws for token in doc)

def main():
    
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The input data file.")
  parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output path of the marked data to be saved.")
  args = parser.parse_args()

  nlp = sp.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
  porter =PorterStemmer()

  bertTokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)
  tsv_writer = open(args.output_path, 'w')

  start_time = time.time()

  print('Counting number of examples...')
  num_lines = sum(1 for line in open(args.data_path, 'r'))
  print('{} examples found.'.format(num_lines))

  with open(args.data_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                time_passed = int(time.time() - start_time)
                print('Processed training set, line {} of {} in {} sec'.format(
                    i, num_lines, time_passed))
                hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to write the training set: {}'.format(
                    hours_remaining))

            idx, query, doc, label = line.rstrip().split()
            q, p = marker(query, doc, nlp, porter)
            tsv_writer.write(f"{idx} {q} {p} {label}\n")
  tsv_writer.close()

if __name__ == "__main__":
    main()