
import torch
from torch.utils.data import Dataset
import linecache
import csv

def encode(tokenizer,  query, 
                       doc, 
                       label, 
                       max_length, 
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
        return torch.tensor(input_ids), torch.tensor(attention_mask),torch.tensor(token_type_ids), torch.tensor(int(label))

class LazyTextDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len):
        self._filename=filename
        with open(filename, "r") as f:
          for i, l in enumerate(f):
            pass
        self._total_data = i
        self.tokenizer=tokenizer
        self._max_len=max_len

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx+2)
        csv_line = csv.reader([line], delimiter=',')
        row= next(csv_line)
        ## Pay attention to the elements' index row[3]= label ...
        return encode(self.tokenizer,row[4],row[5],row[3],self._max_len)  
          
    def __len__(self):
        return self._total_data  