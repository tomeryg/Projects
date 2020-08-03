import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from pathlib import Path


class_num = 10
val_percentage = 0.1

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


def for_val(folder_path, record_file, val_percentage=0.1,class_num=10):
#creates a randomised vector to choose files for validation data
    file_num = 0
    for i in range(class_num):
        folder = i
        data_folder = Path(folder_path + "/%d" %folder)
        directory = os.fsencode(data_folder) 
        for file in os.listdir(directory):
            file_num +=1
    np.random.seed(26)
    to_val = np.random.randint(file_num,size = int(file_num * val_percentage))
    to_val = list(set(to_val))
    to_val = np.sort(to_val)
    lim = len(to_val)-1
    return to_val, lim, file_num

def create_val(folder_path, record_file, class_num=10,val_percentage=0.1):
#creates the validation data tfrecords file
    to_val, lim, _  = for_val(folder_path, record_file, val_percentage, class_num)
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
                    if idx!=lim:
                        idx +=1
                buffer_size +=1

def create_train(folder_path, record_file,class_num= 10, val_percentage=0.1):
#creats the training data tfrecords file
    to_val , lim, _  = for_val(folder_path, record_file, val_percentage, class_num)
    idx = 0
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
                if buffer_size != to_val[idx]:
                    writer_train.write(tf_example.SerializeToString())
                else:
                    if idx!=lim:
                        idx +=1
                buffer_size +=1
    

record_file = 'minst.tfrecords'
folder_path = 'mnist data/trainingSet/trainingSet'
create_val(folder_path, record_file)
create_train(folder_path, record_file)