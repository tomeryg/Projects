import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from pathlib import Path

test = True
record_file = 'minst.tfrecords'
training_folder_path = 'mnist data/trainingSet/trainingSet'
filters = 100
kernels = (3,3)
pools = (3,3)
dense1 = 500
dense2 = 250
last_dense = 10
dropout1 = 0.5
dropout2 = 0.4
epochs = 10


def convert_back(data_type,buffer_size,record_file,val_percentage=0.1, channels =1, img_size = (28,28)):
# converts the tfrecords files to images and labels and returns the parsed dataset
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        # Create a dictionary describing the features.
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        image_features = tf.io.parse_single_example(example_proto,image_feature_description)
        image_buffer = image_features['image_raw']
        image = tf.image.decode_jpeg(image_buffer,channels = channels)
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)*(1. / 255)
        image_shape = tf.stack([img_size[0],img_size[1],channels])
        image = tf.reshape(image,image_shape)
        label = tf.cast(image_features['label'],tf.uint8)
        label = tf.squeeze(label)
        return image,label
    
    batch_size = 32
    num_parallel_batches = 2
    if data_type == 'val':
        buffer = int(buffer_size * val_percentage)
    else:
        buffer = buffer_size
    raw_image_dataset = tf.data.TFRecordDataset(data_type+ '_' + record_file)
    raw_image_dataset = raw_image_dataset.shuffle(buffer)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function, num_parallel_calls = num_parallel_batches)
    parsed_image_dataset = parsed_image_dataset.batch(batch_size)
    parsed_image_dataset = parsed_image_dataset.prefetch(1)
    return parsed_image_dataset



_, _, buffer_size = for_val(training_folder_path, record_file, val_percentage=0.1,numbers=10)
val_ds = convert_back('val',buffer_size = buffer_size, record_file = record_file)
train_ds = convert_back('train',buffer_size = buffer_size, record_file = record_file)

#sanity check - check if your dataset is in the correct shape
if test:
    for image, label in train_ds.take(1):
        print(image.shape, label.shape)
        
#build the cnn model
model = tf.keras.Sequential()
model.add(Conv2D(filters, kernel_size = kernels, padding = 'same', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = pools, padding = 'same'))
model.add(Flatten())
model.add(Dense(dense1, activation = 'relu'))
model.add(Dropout(dropout1))
model.add(Dense(dense2, activation = 'relu'))
model.add(Dropout(dropout2))
model.add(Dense(last_dense, activation = 'softmax'))

model.compile(optimizer = 'adam', metrics = ['sparse_categorical_accuracy'], loss = 'sparse_categorical_crossentropy')

model.fit(train_ds, epochs=10, validation_data = val_ds)




		
		
		

