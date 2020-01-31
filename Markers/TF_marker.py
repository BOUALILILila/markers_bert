import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize as tokenizer
from nltk.corpus import stopwords
import argparse
import string

nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS= set(stopwords.words('english'))
PUNKT=set(string.punctuation)
a='â\x80\x99'
b=a.encode('ISO 8859-1')
b=b.decode('utf-8')
PUNKT.add(b)

porter =PorterStemmer()


## Create the list of dictionaries
def create_struct(text, tokenizer, PUNKT, STOP_WORDS, stemmer):  
  words={}
  for i,word in enumerate(tokenizer(text.lower())):
    if word not in PUNKT and word not in STOP_WORDS:
        if word in words.keys():
          words[word]["indices"].append(i)
        else: 
          obj={"word": word, "indices":[i], "stem": stemmer.stem(word)}
          words[word]=obj
  return words

## Count occurrences in the passages in order to rank query terms w.r.t TF
# Hypothesis: the most frequent query terms are more important
def count_occurrences(match_df):
  match_df['occurrences']=[0]*match_df.shape[0]
  for stem in set(match_df.index.values):
    count=0
    for i,row in match_df.loc[[stem]].iterrows() :
      count+=len(row['indices_y'])
    match_df.loc[[stem],'occurrences']=count

  occur=match_df[['occurrences']].copy(deep=True)
  occur['stem']=occur.index
  occur.drop_duplicates(inplace=True)
  occur.sort_values(by='occurrences',ascending=False, inplace=True)
  occur['rank']=range(occur.shape[0])

  return match_df,occur



def mark(pair):
  try:
      dt=pair['p_txt'].encode("ISO 8859-1")
      doc=dt.decode('utf-8')
  except UnicodeDecodeError:
      doc=pair['p_txt']
        
  try:
      qt=pair['q_txt'].encode("ISO 8859-1")
      query=qt.decode('utf-8')
  except UnicodeDecodeError:
      query=pair['q_txt']

  dt=tokenizer(doc)
  qt=tokenizer(query)

  d_words=create_struct(doc, tokenizer, PUNKT, STOP_WORDS, porter)
  q_words=create_struct(query, tokenizer, PUNKT, STOP_WORDS, porter)

  dfq=pd.DataFrame.from_dict(q_words, orient='index')
  dfd=pd.DataFrame.from_dict(d_words, orient='index')

  match=pd.merge(dfq,dfd,on=['stem'])
  m=match.set_index('stem',drop=True)

  m,occur=count_occurrences(m)

  
  for stem in set(m.index.values):
    for idx, row in m.loc[[stem]].iterrows() :
      id_d=row['indices_y']
      for idd in id_d:
        dt[idd]=f"[e{occur.loc[stem,'rank']}]{dt[idd]}[\e{occur.loc[stem,'rank']}]"
    id_q=row['indices_x']
    for idq in id_q:
        qt[idq]=f"[e{occur.loc[stem,'rank']}]{qt[idq]}[\e{occur.loc[stem,'rank']}]"

  pair['marked_p']=" ".join(dt)
  pair['marked_q']=" ".join(qt)
  pair['stats']=list(occur['occurrences'].values)

  return pair

def main():

  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The input data file.")
  parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output path of the marked data to be saved.")
  args = parser.parse_args()

  dff=pd.read_csv(f"{args.data_path}",index_col=0)

  dff['marked_p']=[0]*len(dff)
  dff['marked_q']=[0]*len(dff)
  dff['stats']=[0]*len(dff)

  dff=dff.apply(mark,axis=1)

  dff.to_csv(f"{args.output_path}")

if __name__ == "__main__":
    main()