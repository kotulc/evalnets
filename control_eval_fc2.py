
import os.path
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# Model parameters as flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 50, 'Number of instances per batch.')
flags.DEFINE_integer('max_steps', 30000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('conv_fmaps', 16, 'Number of feature maps (channels).')
flags.DEFINE_integer('fc1_nodes', 500, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('fc2_nodes', 100, 'Number of units in hidden layer 3.')
flags.DEFINE_float('keep_prob', 0.5, 'Output dropout probability.')
flags.DEFINE_string('train_dir','data','Directory to store training logs')


# Return a weight variable initialized with the given shape
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

# Return a bias variable initialized with the given shape
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

# Return a convolution of x and W with 2x2 stride
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

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
    # Download data if no local copy exists
    data_sets = input_data.read_data_sets(FLAGS.train_dir, one_hot=True)

    # Create the session
    sess = tf.InteractiveSession()

    # Input and label placeholders
    x = tf.placeholder('float', shape=[None, 784], name='x-input')
    y_ = tf.placeholder('float', shape=[None, 10], name='y-input')
    keep_prob = tf.placeholder('float', name='k-prob')


    # Convolutional layer - variables
    with tf.name_scope('conv'):
        W_conv = weight_variable([4, 4, 1, FLAGS.conv_fmaps])
        b_conv = bias_variable([FLAGS.conv_fmaps])

        # Reshape and convolve
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)


    # Fully connected layer1 - variables
    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([7 * 7 * FLAGS.conv_fmaps, FLAGS.fc1_nodes])
        b_fc1 = bias_variable([FLAGS.fc1_nodes])

        # Reshape and apply relu
        h_pool1_flat = tf.reshape(h_pool, [-1, 7 * 7 * FLAGS.conv_fmaps])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        # If the max_pool operation is ignored...
        #h_flat = tf.reshape(h_conv, [-1, 14 * 14 * FLAGS.conv_fmaps])
        #h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)


    # Fully connected layer2 - variables
    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([FLAGS.fc1_nodes, FLAGS.fc2_nodes])
        b_fc2 = bias_variable([FLAGS.fc2_nodes])

        # Apply relu
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # Apply dropout to output
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Readout layer
    with tf.name_scope('readout'):
        W_out = weight_variable([FLAGS.fc2_nodes, 10])
        b_out = bias_variable([10])

        y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_out) + b_out)


    # Add summary ops for tensorboard
    _ = tf.histogram_summary('W_conv', W_conv)
    _ = tf.histogram_summary('W_fc1', W_fc1)
    _ = tf.histogram_summary('W_fc2', W_fc2)
    _ = tf.histogram_summary('W_out', W_out)
    _ = tf.histogram_summary('b_conv', b_conv)
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
            # Generate a new feed dictionary
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
            # Generate a new feed dictionary
            feed_dict = fill_feed_dict(
                    data_sets.train, x, y_, (keep_prob, FLAGS.keep_prob))
            sess.run(train_step, feed_dict=feed_dict)

    print('test accuracy %.3f'%accuracy.eval(feed_dict={
        x: data_sets.test.images, y_: data_sets.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run()