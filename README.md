# markers_bert

Investigate the 

## First Stage: Doc2query Passage Expansion + BM25
We use the traditional BM25 to retrieve an intital list of the top 1000 passages per query. To avoid the "vocabulary mismatch" problem we apply the Doc2query passage expansion technique by [(Nogueira et al., 2019)](https://arxiv.org/pdf/1904.08375.pdf). [Here is the link to the github repo](https://github.com/nyu-dl/dl4ir-doc2query).

## Second Stage: Enhanced BERT re-ranking
### Data preparation
First we need to put MsMarco data in the appropriate format. 
- For the fine-tuning use the script below to create the .csv file of unique pairs from the ```triples.train.small.tsv``` file: 
```
python construct_train_data.py \
          -
```

- For the inference data. Once you get the run file from the first stage ([download here](https://drive.google.com/file/d/1uW2JF5aXDTjlKUnMQttXrCPo5pqjEphk/view?usp=sharing)). Use the script below :
```
python construct_train_data.py \
          -
```
This script produces two files the first is the the data file ```dataset.csv``` and the second is the ```query_doc_ids.txt```. 
### Base Model
Basic BERT-base model re-ranker that uses \[CLS\] token for classification. Use the script below to fine-tune this model and evaluate it:
```
python ./Modeling/modeling_base.py \
      --data_dir=$DATA_DIR \
      --output_dir=$OUTPUT_DIR \
      --max_seq_length=512 \
      --do_train \
      --do_eval \
      --per_gpu_eval_batch_size=32 \
      --per_gpu_train_batch_size=32 \
      --gradient_accumulation_steps=1 \
      --learning_rate=3e-6 \
      --weight_decay=0.01 \
      --adam_epsilon=1e-8 \
      --max_grad_norm=1.0 \
      --num_train_epochs=2 \
      --warmup_steps=10398 \
      --logging_steps=1000 \
      --save_steps=25996 \
      --seed=42 \
      --local_rank=-1 \
      --overwrite_output_dir
```

### Strategy 1: Incorporating Exact Match signals via Markers
1. We first need to mark both the training dataset and the dev set using this script: 
```
python ./Markers/simple_marker.py \
      --data_path=$path_to_dataset.csv
      --output_path=$path_to_marked_data
```
2. Fine-tune the BERT-base model using the ```marked```  data and evaluate it on the ```marked``` dev set :
```
python ./Modeling/modeling_markers.py \
      --data_dir=$DATA_DIR \
      --output_dir=$OUTPUT_DIR \
      --max_seq_length=512 \
      --do_train \
      --do_eval \
      --do_lower_case\
      --per_gpu_eval_batch_size=32 \
      --per_gpu_train_batch_size=32 \
      --gradient_accumulation_steps=1 \
      --learning_rate=3e-6 \
      --weight_decay=0.01 \
      --adam_epsilon=1e-8 \
      --max_grad_norm=1.0 \
      --num_train_epochs=4 \
      --warmup_steps=10398 \
      --logging_steps=1000 \
      --save_steps=25996 \
      --seed=42 \
      --local_rank=-1 \
      --overwrite_output_dir
```

### Strategy 2: Incorporating Exact Match + TF importance signals 
1. We first need to mark both the training dataset and the dev set using this script: 
```
python ./Markers/TF_marker.py \
      --data_path=$path_to_dataset.csv
      --output_path=$path_to_marked_data
```
2. Fine-tune the BERT-base model using the ```marked```  data and evaluate it on the ```marked``` dev set :
```
python ./Modeling/modeling_TF.py \
      --data_dir=$DATA_DIR \
      --output_dir=$OUTPUT_DIR \
      --K=2 \
      --max_seq_length=512 \
      --do_train \
      --do_eval \
      --do_lower_case\
      --per_gpu_eval_batch_size=32 \
      --per_gpu_train_batch_size=32 \
      --gradient_accumulation_steps=1 \
      --learning_rate=3e-6 \
      --weight_decay=0.01 \
      --adam_epsilon=1e-8 \
      --max_grad_norm=1.0 \
      --num_train_epochs=4 \
      --warmup_steps=10398 \
      --logging_steps=1000 \
      --save_steps=25996 \
      --seed=42 \
      --local_rank=-1 \
      --overwrite_output_dir
```


