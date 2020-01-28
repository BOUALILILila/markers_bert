import torch


import linecache
import csv


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
