"""
Trains a network with the fc1-fc2-out architecture against MNIST train_data
filtered through an external convolutional layer. Utilizes input_data function
to read the matlab data file containing the output of this external conv layer.

Note: Structured similary to control_eval_fc2, however, the parameter file_path
is not set, the network is trained against the original MNIST data instead
"""

import os.path
import tensorflow as tf

#from tensorflow.examples.tutorials.mnist import input_data
import input_data

# Model parameters as flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 50, 'Number of instances per batch.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.5, 'Output dropout probability.')
flags.DEFINE_integer('conv_fmaps', 6, 'Number of feature maps (channels).')
flags.DEFINE_integer('fc1_nodes', 600, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('fc2_nodes', 200, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('target_label', 0, 'Target label for binary one-hot.')
flags.DEFINE_string('train_dir', 'data', 'Directory to store training logs.')
flags.DEFINE_string('file_path', 'data/features.m', 'Matlab feature data path')


# Return a weight variable initialized with the given shape
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')


# Return a bias variable initialized with the given shape
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')


# Return a 2x2 max pool layer with 2x2 stride
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')


# fill_feed_dict from fully_connected_feed.py
def fill_feed_dict(data_sets, x, y_, keep_tuple):
  x_feed, y_feed = data_sets.next_batch(FLAGS.batch_size, False)
  feed_dict = { x: x_feed, y_: y_feed, keep_tuple[0] : keep_tuple[1]}
  return feed_dict


def main(_):

    # If the matlab file_path is not set, load raw MNIST data
    if not FLAGS.file_path:
        # Download data if no local copy exists
        data_sets = input_data.read_data_sets(FLAGS.train_dir, one_hot=True,
                                      target_label=FLAGS.target_label)
    else:
        # Read from matlab data file if passed file_path
        data_sets = input_data.read_mdata_sets(FLAGS.file_path, one_hot=True,
                                      target_label=FLAGS.target_label)


    # Create the session
    sess = tf.InteractiveSession()

    # Input and label placeholders
    num_classes = data_sets.train.num_classes
    num_features = data_sets.train.num_features
    x = tf.placeholder('float', shape=[None, num_features], name='x-input')
    y_ = tf.placeholder('float', shape=[None, num_classes], name='y-input')
    keep_prob = tf.placeholder('float', name='k-prob')


    # Input layer - operations
    #with tf.name_scope('input'):
        # Reshape and convolve
        #x_image = tf.reshape(x, [-1, 14, 14, 1])
        #h_pool = max_pool_2x2(x_image)


    # Fully connected layer1
    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([num_features, FLAGS.fc1_nodes])
        b_fc1 = bias_variable([FLAGS.fc1_nodes])

        # Reshape and apply relu
        #h_pool1_flat = tf.reshape(h_pool, [-1, num_features])
        #h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        # If the max_pool operation is ignored...
        #h_flat = tf.reshape(h_conv, [-1, 14 * 14 * FLAGS.conv_fmaps])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)


    # Fully connected layer2
    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([FLAGS.fc1_nodes, FLAGS.fc2_nodes])
        b_fc2 = bias_variable([FLAGS.fc2_nodes])

        # Apply relu
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # Apply dropout to fc_2 output
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Readout layer
    with tf.name_scope('readout'):
        W_out = weight_variable([FLAGS.fc2_nodes, num_classes])
        b_out = bias_variable([num_classes])

        y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_out) + b_out)


    # Add summary ops for tensorboard
    _ = tf.histogram_summary('W_fc1', W_fc1)
    _ = tf.histogram_summary('W_fc2', W_fc2)
    _ = tf.histogram_summary('W_out', W_out)
    _ = tf.histogram_summary('b_fc1', b_fc1)
    _ = tf.histogram_summary('b_fc2', b_fc2)
    _ = tf.histogram_summary('b_out', b_out)
    _ = tf.histogram_summary('Output', y)


    # Cost function
    with tf.name_scope('xent'):
        x_entropy = -tf.reduce_sum(y_ * tf.log(y))
        _ = tf.scalar_summary('xentropy', x_entropy)

    # Train the model
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(x_entropy)

    # Evaluate model
    with tf.name_scope('eval'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        _ = tf.scalar_summary('accuracy', accuracy)

    # Collect all summaries during graph building
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)

    sess.run(tf.initialize_all_variables())


    # Train the model and record summaries
    for i in range(FLAGS.max_steps):
        if i%50 == 0:
            # Generate a new feed dictionary to test training accuracy
            feed_dict = fill_feed_dict(
                    data_sets.train, x, y_, (keep_prob, 1.0))
            # Update the summary collection
            result = sess.run([summary_op, accuracy], feed_dict=feed_dict)
            summary_str = result[0]
            summary_writer.add_summary(summary_str, i)
            train_accuracy = result[1]
            # Print status update
            print('step %d, training accuracy %g'%(i, train_accuracy))
        else:
            # Generate a new feed dictionary for the next training batch
            feed_dict = fill_feed_dict(
                    data_sets.train, x, y_, (keep_prob, FLAGS.keep_prob))
            sess.run(train_step, feed_dict=feed_dict)

    print('test accuracy %.4f'%accuracy.eval(feed_dict={
        x: data_sets.test.images, y_: data_sets.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run()