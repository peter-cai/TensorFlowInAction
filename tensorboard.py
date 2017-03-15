import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
max_steps = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = 'MNIST_data/'
log_dir = 'logs/mnist_with_summaries'
mnist = input_data.read_data_sets(data_dir, one_hot = True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name = 'x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name = 'y-output')

with tf.name_scope('input_reshape'):
  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', image_shaped_input, 10)

# creat random matrix with normal distrubution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# create constant matrix
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('sttdev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name = 'activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

with tf.name_scope('cross_entropy'):
  diff =  tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_)
  with tf.name_scope('total'):
    cross_entropy = -tf.reduce_mean(diff)
tf.summary.scalar('cross entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('cross entropy', cross_entropy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir +  '/test')
tf.global_variables_initializer().run()

# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100)
    k = dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

saver = tf.train.Saver()
for i in range(max_steps):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    if i % 100 == 99:  # Record execution stats
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      saver.save(sess, log_dir + '/modle.ckpt', i)
      print('Adding run metadata for', i)
    else:  # Record a summary
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()

# Finally, open a terminal imput:
# $ tensorboard --logdir = /tmp/tensorflow/logs/mnist_with_summaries
# result: 
# Starting TensorBoard b'39' on port 6006
# (You can navigate to http://192.168.233.101:6006)
# Open the url above in the chrome (Only in Linux)