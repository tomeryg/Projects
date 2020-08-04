import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from pathlib import Path


class_num = 10
val_percentage = 0.1
test_percentage = 0.1

#helper functions for the features.
def _int64_feature(value):
#Wrapper for inserting int64 features into Example proto.
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
#Wrapper for inserting bytes features into Example proto.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def image_example(image_string, label):
#creates the example to write in the TFrecords file
    image_shape = tf.image.decode_image(image_string)
    feature = {
      'height': _int64_feature(image_shape.shape[0]),
      'width': _int64_feature(image_shape.shape[1]),
      'depth': _int64_feature(image_shape.shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def randomize_data(folder_path, record_file, val_percentage = 0.1, test_percentage = 0.1, class_num = 10):
#creates a randomised vector to choose files for validation and test data
    file_num = 0
    for i in range(class_num):
        folder = i
        data_folder = Path(folder_path + "/%d" %folder)
        directory = os.fsencode(data_folder) 
        for file in os.listdir(directory):
            file_num +=1
    np.random.seed(126)
    randomized = np.random.choice(file_num, int(file_num * (val_percentage + test_percentage)), replace = False)
    to_val = np.sort(np.random.choice(randomized, int(file_num * val_percentage), replace = False))
    to_test = np.sort([x for x in randomized if x not in to_val])
    lim_val = len(to_val) - 1
    lim_test = len(to_test) - 1 
    return to_val, lim_val, to_test, lim_test, file_num

def create_val(folder_path, record_file, to_val, lim_val, class_num = 10):
#creates the validation data tfrecords file
    idx = 0
    buffer_size=0
    file_name = 'val_' + record_file
    with tf.io.TFRecordWriter(file_name) as writer_val:
        for i in range(class_num):
            folder = i
            data_folder = Path(folder_path + "/%d" %folder)
            directory = os.fsencode(data_folder) 
            for file in os.listdir(directory):
                filename = data_folder / os.fsdecode(file)
                image_string = open(filename,'rb').read()
                label = folder
                tf_example = image_example(image_string, label)
                if buffer_size == to_val[idx]:
                    writer_val.write(tf_example.SerializeToString())
                    if idx!=lim_val:
                        idx +=1
                buffer_size +=1

def create_train(folder_path, record_file, to_val, lim_val,to_test, lim_test, class_num = 10):
#creats the training data tfrecords file
    idx_val = 0
    idx_test = 0
    buffer_size=0
    file_name = 'train_' + record_file                    
    with tf.io.TFRecordWriter(file_name) as writer_train:
        for i in range(class_num):
            folder = i
            data_folder = Path(folder_path + "/%d" %folder)
            directory = os.fsencode(data_folder) 
            for file in os.listdir(directory):
                filename = data_folder / os.fsdecode(file)
                image_string = open(filename,'rb').read()
                label = folder
                tf_example = image_example(image_string, label)
                if buffer_size != to_val[idx_val] and buffer_size!=to_test[idx_test]:
                    writer_train.write(tf_example.SerializeToString())
                elif buffer_size == to_val[idx_val]:
                    if idx_val!=lim_val:
                        idx_val +=1
                else:
                    if idx_test!=lim_test:
                        idx_test +=1
                buffer_size +=1

def create_test(folder_path, record_file, to_test, lim_test, class_num = 10, test_percentage = 0.1):
#creats the test data tfrecords file
    idx = 0
    buffer_size=0
    file_name = 'test_' + record_file                    
    with tf.io.TFRecordWriter(file_name) as writer_test:
        for i in range(class_num):
            folder = i
            data_folder = Path(folder_path + "/%d" %folder)
            directory = os.fsencode(data_folder) 
            for file in os.listdir(directory):
                filename = data_folder / os.fsdecode(file)
                image_string = open(filename,'rb').read()
                label = folder
                tf_example = image_example(image_string, label)
                if buffer_size == to_test[idx]:
                    writer_test.write(tf_example.SerializeToString())
                    if idx!=lim_test:
                        idx +=1
                buffer_size +=1                
        
                
        

record_file = 'minst.tfrecords'
folder_path = 'mnist data/trainingSet/trainingSet'
to_val, lim_val , to_test, lim_test = randomize_data(folder_path, record_file, val_percentage, test_percentage ,class_num)
create_val(folder_path, record_file, to_val, lim_val, class_num)
create_train(folder_path, record_file, to_val, lim_val,to_test, lim_test, class_num)
create_test(folder_path, record_file, to_test, lim_test, class_num, test_percentage)
