import nltk

nltk.download('punkt')
nltk.download('stopwords')

code_to_test="""
import pandas as pd
data="/projets/iris/PROJETS/lboualil/workdata/msmarco-passage/top1000.doc2query.dev.small_full_ids_free.csv"
dff= pd.read_csv(data, index_col=0)
#dff=dff.reset_index(drop=True)

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize as tokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re, string
import json
import copy

STOP_WORDS= set(stopwords.words('english'))
PUNKT=set(string.punctuation)
a='â\x80\x99'
b=a.encode('ISO 8859-1')
b=b.decode('utf-8')
PUNKT.add(b)

porter =PorterStemmer()

def mark(pair):
  try:
        dt=pair['p_txt'].encode("ISO 8859-1")
        dt=dt.decode('utf-8')
  except UnicodeDecodeError:
        dt=pair['p_txt']
  
  try:
        qt=pair['q_txt'].encode("ISO 8859-1")
        qt=qt.decode('utf-8')
  except UnicodeDecodeError:
        qt=pair['q_txt']
  
  
  d=tokenizer(dt)
  q=tokenizer(qt)
  ql=copy.deepcopy(q)
  i=0
  sq=set()
  for word in ql:
    stem= porter.stem(word)
    if word not in PUNKT and word.lower() not in STOP_WORDS and stem not in sq:
      
      match=0
      for idd,token in enumerate(d):
        if porter.stem(token) ==stem:
          d[idd]=f"[e{i}]{token}[\e{i}]"
          match=1
      if match:
        for idq,token in enumerate(q):
          if porter.stem(token) ==stem:
            q[idq]=f"[e{i}]{token}[\e{i}]"
        i+=1
      sq.add(stem)
      
  pair['marked_q'] = " ".join(q)
  pair['marked_p'] = " ".join(d)
  return pair

dff['marked_q']=[0]*len(dff)
dff['marked_p']=[0]*len(dff)
dff=dff.apply(mark,axis=1)
dff.to_csv('/projets/iris/PROJETS/lboualil/workdata/msmarco-passage/doc2query_run/markers/marked_top1000.doc2query.dev.small_full_ids_free.csv')
"""
import timeit
elapsed_time = timeit.timeit(code_to_test, number=1)
print(elapsed_time)