import os, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import math
from PIL import Image
from math import ceil
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary#, print_hist_summery, get_hist, per_class_acc, writeImage
from inputs import *
from functools import reduce
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "8", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', ".01", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "logs", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('trn_im_dir', "bkg_static/ambient", """ path to images """)
tf.app.flags.DEFINE_string('trn_la_dir', "Im_la/ambient", """ path to image labe """)
tf.app.flags.DEFINE_string('tst_im_dir', "bkg_static/add_blue_light", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('tst_la_dir', "Im_la/add_blue_light", """ path to CamVid val image """)

tf.app.flags.DEFINE_integer('max_steps', "2000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "112", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "112", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "1", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "1", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)



# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1 # Learning rate decay factor.
decay_steps =1000 #int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)


BATCH_SIZE = 8
NUM_CLASSES = 1


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)   #OR tf.random_normal 


def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


def loss(predictions, labels):

    Loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predictions,labels))))
    tf.add_to_collection('losses', Loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def conv_layer_with_bn(inputT, shape, train_phase, name=None, activation_fn=tf.nn.relu):
    #shape = [k_size, k_size, in_channel, out_channel]
    out_channel = shape[3]

    with tf.variable_scope(name) as scope:
      kernel = _variable_with_weight_decay('weights', shape=shape, initializer=orthogonal_initializer(), wd=None)   #orthogonal_initializer() or xaviar initialization
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      
      #batch_norm = batch_norm_layer(bias, train_phase, scope.name)
      #conv_out = tf.nn.relu(batch_norm)
      conv_out = tf.nn.relu(bias)
      
      _activation_summary(inputT)

    return conv_out

def fully_connected(inputT, name, out_channel, activation_fn=tf.nn.relu):
    in_channel = inputT.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        kernel = _variable_with_weight_decay('weights', shape = [in_channel, out_channel], initializer=orthogonal_initializer(),wd=0.0005)  #tf.contrib.layers.xavier_initializer() or msra_initializer(3, in_channel)
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0)) 
        out = tf.nn.bias_add(tf.matmul(inputT, kernel), biases)
        out = activation_fn(out)
        _activation_summary(out)
    return out

#IMPORTANT note:
#the impoove the performance on validation and test set, tune the parameters in tf.contrib.layers.batch_norm

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def inference(images, phase_train):
    # assuming 224x224x3 input_tensor
    keep_prob = 0.75
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    print('layer 1 dimension',norm1.get_shape(), images.get_shape())

    # block 1 -- outputs 112x112x64

    conv1_1 = conv_layer_with_bn(norm1, [3, 3, images.get_shape().as_list()[3], 64], phase_train, name="conv1_1") #64
    conv1_2 = conv_layer_with_bn(conv1_1, [3, 3, conv1_1.get_shape().as_list()[3], 64], phase_train, name="conv1_2")
    #conv1_2 = tf.nn.lrn(conv1_2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_layer_with_bn(pool1, [3, 3, pool1.get_shape().as_list()[3], 128], phase_train, name="conv2_1")
    #conv2_1 = tf.nn.lrn(conv2_1, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    conv2_2 = conv_layer_with_bn(conv2_1, [3, 3, conv2_1.get_shape().as_list()[3], 128], phase_train, name="conv2_2")
    #conv2_2 = tf.nn.lrn(conv2_2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')
    #net = tf.nn.dropout(pool2, keep_prob)

    # block 3 -- outputs 56x56x128
    conv3_1 = conv_layer_with_bn(pool2, [3, 3, pool2.get_shape().as_list()[3], 128], phase_train, name="conv3_1")
    #conv3_1 = tf.nn.lrn(conv3_1, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    conv3_2 = conv_layer_with_bn(conv3_1, [3, 3, conv3_1.get_shape().as_list()[3], 128], phase_train, name="conv3_2")
    #conv3_2 = tf.nn.lrn(conv3_2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool3')
    #net = tf.nn.dropout(pool3, keep_prob)
    


    #block 4 -- outputs 14x14x512
    conv4_1 = conv_layer_with_bn(pool3, [3, 3, pool3.get_shape().as_list()[3], 512], phase_train, name="conv4_1")
    #conv4_1 = tf.nn.lrn(conv4_1, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    conv4_2 = conv_layer_with_bn(conv4_1, [3, 3, conv4_1.get_shape().as_list()[3], 512], phase_train, name="conv4_2")
    #conv4_2 = tf.nn.lrn(conv4_2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    conv4_3 = conv_layer_with_bn(conv4_2, [3, 3, conv4_2.get_shape().as_list()[3], 512], phase_train, name="conv4_3")
    #conv4_3 = tf.nn.lrn(conv4_3, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool4')
    #net = tf.nn.dropout(net, keep_prob)
    
    
    
    # block 5 -- outputs 7x7x512
    conv5_1 = conv_layer_with_bn(pool4, [3, 3, pool4.get_shape().as_list()[3], 512], phase_train, name="conv5_1")
    #conv5_2 = tf.nn.lrn(conv5_1, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    conv5_2 = conv_layer_with_bn(conv5_1, [3, 3, conv5_1.get_shape().as_list()[3], 512], phase_train, name="conv5_2")
    #conv5_2 = tf.nn.lrn(conv5_2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    conv5_3 = conv_layer_with_bn(conv5_2, [3, 3, conv5_2.get_shape().as_list()[3], 512], phase_train, name="conv5_3")
    #conv5_3 = tf.nn.lrn(conv5_3, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool5')
    #net = tf.nn.dropout(net, keep_prob)
    print('layer 5 dimension',pool5.get_shape())

    # flatten
    flat_shape = reduce(lambda x,y:x*y,pool5.get_shape().as_list()[1:])
    net = tf.reshape(pool5, [-1, flat_shape], name="flatten")
    print('flatten layer',net.get_shape())

    # fully connected
     net = fully_connected(net, name="fc6", out_channel=1024)
    net = tf.nn.dropout(net, keep_prob)
    net = fully_connected(net, name="fc7", out_channel=1024)   
    net = tf.nn.dropout(net, keep_prob)


    predictions = fully_connected(net, name="fc8", out_channel=FLAGS.num_class)


    return predictions


'''def train(total_loss, global_step):

   
    """Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps processed.
    Returns:
       train_op: op for training.
    """
 
    """ fix lr """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                     global_step, decay_steps,
                                     LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate',lr)
  
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr) 
      #opt = tf.train.GradientDescentOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op
'''


def training(FLAGS, is_finetune=False):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  initial_lr = FLAGS.learning_rate
  trn_im_dir = FLAGS.trn_im_dir
  trn_la_dir = FLAGS.trn_la_dir 
  tst_im_dir = FLAGS.tst_im_dir
  tst_la_dir = FLAGS.tst_la_dir 
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c

  startstep = 0 

  # For data preparation
  trn_im_data, val_im_data,trn_labels, val_labels = data_preprocessing(trn_im_dir, trn_la_dir, phase='train')
  #trn_im_data = tf.image.resize_images(trn_im_data, [image_h, image_w])
  #val_im_data = tf.image.resize_images(val_im_data, [image_h, image_w])
  #print('$$$$$$$$$$$$$$$$$$$$$$$$$',val_im_data.get_shape())
  tst_im_data, tst_labels = data_preprocessing(tst_im_dir, tst_la_dir, select=200, phase='test')
  #tst_im_data = tf.image.resize_images(tst_im_data, [image_h, image_w])
  '''r, g, b = np.mean(trn_im_data[:,:,:,0]),np.mean(trn_im_data[:,:,:,1]),np.mean(trn_im_data[:,:,:,2])
  trn_im_data[:,:,:,0] = trn_im_data[:,:,:,0] - r
  trn_im_data[:,:,:,1] = trn_im_data[:,:,:,1] - g
  trn_im_data[:,:,:,2] = trn_im_data[:,:,:,2] - b
  tst_im_data[:,:,:,0] = tst_im_data[:,:,:,0] - r
  tst_im_data[:,:,:,1] = tst_im_data[:,:,:,1] - g
  tst_im_data[:,:,:,2] = tst_im_data[:,:,:,2] - b'''
        
    
    
  with tf.Graph().as_default():

    X = tf.placeholder( tf.float32, shape=[None, image_h, image_w, image_c])
    Y = tf.placeholder(tf.float32, shape=[None])
    
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    global_step = tf.Variable(0, trainable=False)

    predictions = inference(X, phase_train)
    Loss = loss(predictions, trn_labels)
    #lr = tf.train.exponential_decay(initial_lr, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR)
    lr = initial_lr
    optimizer = tf.train.AdamOptimizer(lr).minimize(Loss)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        total_batches = len(trn_im_data) // batch_size
        #val_im_data = tf.image.resize_images(val_im_data, [image_h, image_w])
        #val_im_data = sess.run(val_im_data)
        tst_im_data = tf.image.resize_images(tst_im_data, [image_h, image_w])
        tst_im_data = sess.run(tst_im_data)
        for step in range(max_steps):
            total_train_loss = 0
            for b in range(total_batches):
                offset = (b * batch_size) % (len(trn_im_data) - batch_size) 
                #plt.imshow(image_batch[0])
                #plt.show()
                batch_x = trn_im_data[offset:(offset + batch_size), :, :,:]
                batch_x = tf.image.resize_images(batch_x, [image_h, image_w])
                batch_x = sess.run(batch_x)
                batch_y = trn_labels[offset:(offset + batch_size)]
                feed_dict = {X: batch_x, Y: batch_y, phase_train: True}
                
                _, loss_value = sess.run([optimizer, Loss], feed_dict=feed_dict)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                total_train_loss += loss_value
            

                
            if step % 10 == 0:
                print(step, 'total_train_loss:',total_train_loss/total_batches)
             
                '''pred = sess.run(predictions, feed_dict={X:val_im_data,phase_train:'False'})
                #total_val_loss = loss(pred, val_labels)
                total_val_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred,val_labels))))
                print ('val_losss: ',sess.run(total_val_loss))'''
                
                pred = sess.run(predictions, feed_dict={X:tst_im_data,phase_train:'False'})
                #total_val_loss = loss(pred, val_labels)
                total_tst_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred,tst_labels))))
                print ('test_losss: ',sess.run(total_tst_loss))
                
                print('-------------------------------------------------------\n')
                print(list(zip(tst_labels, pred.tolist())))
                
                
                
training(FLAGS)







