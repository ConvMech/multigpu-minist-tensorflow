# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/guide/reading_data#reading_from_files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def decode(serialized_example):
  """Parses an image and label from the given `serialized_example`."""
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape((mnist.IMAGE_PIXELS))

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def augment(image, label):
  """Placeholder for data augmentation."""
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.
  return image, label


def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).

    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
  """
  if not num_epochs:
    num_epochs = None
  filename = os.path.join(FLAGS.train_dir, TRAIN_FILE
                          if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    dataset = dataset.map(decode)
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)

    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    dataset = dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def model(X, reuse=False):
    with tf.variable_scope('L1', reuse=reuse):
        L1 = tf.layers.conv2d(X, 64, [3, 3], reuse=reuse)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, 0.7, True)

    with tf.variable_scope('L2', reuse=reuse):
        L2 = tf.layers.conv2d(L1, 128, [3, 3], reuse=reuse)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, 0.7, True)

    with tf.variable_scope('L2-1', reuse=reuse):
        L2_1 = tf.layers.conv2d(L2, 128, [3, 3], reuse=reuse)
        L2_1 = tf.layers.max_pooling2d(L2_1, [2, 2], [2, 2])
        L2_1 = tf.layers.dropout(L2_1, 0.7, True)

    with tf.variable_scope('L3', reuse=reuse):
        L3 = tf.contrib.layers.flatten(L2_1)
        L3 = tf.layers.dense(L3, 1024, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, 0.5, True)

    with tf.variable_scope('L4', reuse=reuse):
        L4 = tf.layers.dense(L3, 256, activation=tf.nn.relu)

    with tf.variable_scope('LF', reuse=reuse):
        LF = tf.layers.dense(L4, 10, activation=None)

    return LF

def inference(images, hidden1_units, hidden2_units,reuse=False):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  import math
  NUM_CLASSES = 10
  IMAGE_SIZE = 28
  IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

  ''' 
  #correct way! using variablee scope when building graph only!"
  # Hidden 1
  with tf.variable_scope('hidden1'):
    weights = tf.get_variable('weights',[IMAGE_PIXELS, hidden1_units])
    biases = tf.get_variable('biases',[hidden1_units])
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    
  # Hidden 2
  with tf.variable_scope('hidden2'):
    weights = tf.get_variable('weights',[hidden1_units, hidden2_units])
    biases = tf.get_variable('biases',[hidden2_units])
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.variable_scope('softmax_linear'):
    weights = tf.get_variable('weights',[hidden2_units, NUM_CLASSES])
    biases = tf.get_variable('biases',[NUM_CLASSES])
    logits = tf.matmul(hidden2, weights) + biases
  '''
  
  with tf.variable_scope('model',reuse=reuse):
      weights = tf.get_variable('weights_1',[IMAGE_PIXELS, hidden1_units])
      biases = tf.get_variable('biases_1',[hidden1_units])
      hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

      weights = tf.get_variable('weights_2',[hidden1_units, hidden2_units])
      biases = tf.get_variable('biases_2',[hidden2_units])
      hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

      weights = tf.get_variable('weights_3',[hidden2_units, NUM_CLASSES])
      biases = tf.get_variable('biases_3',[NUM_CLASSES])
      logits = tf.matmul(hidden2, weights) + biases
    
  return logits

def run_training():
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    # create optimizer and global step
    global_step = tf.Variable(0, name='global_step',trainable=False)
    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    # prepare the training data batches
    image_batch, label_batch = inputs(
        train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    # prepare the gradiant towers
    tower_grads = []

    # now start building towers
    
    print("********",image_batch)
        
    tower_image_batch = tf.split(image_batch, 2)
    tower_label_batch = tf.split(label_batch, 2)
    
    print("#"*50)
    scope = tf.get_variable_scope()
    print(scope.name)
    print("#"*50)
   
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(2):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ("tower", i)) as scope:

            # create the model on each gpu
            logits = inference(tower_image_batch[i], FLAGS.hidden1, FLAGS.hidden2,reuse=(i>0))

            # get the loss from each gpu
            loss = tf.losses.sparse_softmax_cross_entropy(labels=tower_label_batch[i], logits=logits,scope=scope)
            #loss = mnist.loss(logits, tower_label_batch[i])

            # Reuse variables for the next tower.
            #tf.get_variable_scope().reuse_variables()

            # Calculate the gradients for the batch of data on this CIFAR tower.
            #grads = opt.compute_gradients(loss,tf.trainable_variables(scope=scope))
            
            #grads = opt.compute_gradients(loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
            grads = opt.compute_gradients(loss)
                                          
            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    
    
    for i in tower_grads:
        print('*'*50)
        for j in i:
            print(j)
        
    grads = average_gradients(tower_grads)

    #grads = tower_grads[0]
    #grad1 = tower_grads[0]
    #grad2 = tower_grads[1]
    #grad_mean = grads
    
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    MOVING_AVERAGE_DECAY = 0.9999

    #variable_averages = tf.train.ExponentialMovingAverage(
    #    MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    #train_op = tf.group(apply_gradient_op, variables_averages_op)
    
    train_op = apply_gradient_op
    
    # now back to the regular
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)) as sess:
        
      # Initialize the variables (the trained variables and the
      # epoch counter).
      sess.run(init_op)
      writer = tf.summary.FileWriter(".", sess.graph)
      try:
        step = 0
        while True:  # Train until OutOfRangeError
          start_time = time.time()

          # Run one step of the model.  The return values are
          # the activations from the `train_op` (which is
          # discarded) and the `loss` op.  To inspect the values
          # of your ops or variables, you may include them in
          # the list passed to sess.run() and the value tensors
          # will be returned in the tuple from the call.
        
          #g1,g2,gm = sess.run([grad1,grad2,grad_mean])
          #label = sess.run([label_batch])
          #_, loss_value,g1,g2,gm = sess.run([train_op, loss,grad1,grad2,grad_mean])
          _, loss_value = sess.run([train_op, loss])
        
          duration = time.time() - start_time
            
          # Print an overview fairly often.
          if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                       duration))
            #print("1",g1[0])
            #print("2",g2[0])
            #print("m",gm[0])
            #print(len(label[0]))
            
          step += 1
      except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                          step))

def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.')
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=2,
      help='Number of epochs to run trainer.')
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.')
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.')
  parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./data',
      help='Directory with the training data.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
