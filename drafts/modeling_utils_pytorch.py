import tensorflow as tf

def extract_fn(data_record):  
    features = {
          "input_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "attention_mask": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "token_type_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
      }
    sample = tf.io.parse_single_example(data_record, features)
      
    input_ids = tf.cast(sample["input_ids"], tf.int32)
    token_type_ids = tf.cast(sample["token_type_ids"], tf.int32)
    label_ids = tf.cast(sample["label"], tf.int32)
    attention_mask = tf.cast(sample["attention_mask"], tf.int32)
      
    features = (
          input_ids,
          attention_mask,
          token_type_ids,
          label_ids
      )
    
    return features


def get_dataset(dataset_path, batch_size, seq_length, is_training_set=False):
    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(
                extract_fn, num_parallel_calls=4).prefetch(1000*batch_size)
    if is_training_set:
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=(
                    [seq_length],
                    [seq_length],
                    [seq_length],
                    []                    
                ),
                padding_values=(0,0,0,0),
                drop_remainder=True)
    return dataset

