import input_data
from input_data import create_datasets
import numpy as np
import tensorflow as tf

NUM_IMAGES = 500

classes = np.array([["dog", "n02084071"],
                    ["cat", "n02121808"],
                    ["bird", "n01503061"],
                    ["orange", "n07747607"],
                    ["apple", "n07739125"],
                    ["keyboard", "n03614007"],
                    ["computer mouse", "n03793489"],
                    ["desk", "n03179701"],
                    ["monitor", "n03782006"],
                    ["book", "n02870526"],
                    ["pen", "n03906997"],
                    ["pencil", "n03908204"],
                    ["chair", "n03001627"],
                    ["sword", "n04373894"],
                    ["cup", "n03147509"],
                    ["shirt", "n04197391"],
                    ["shoe", "n04200000"],
                    ["car", "n02960352"],
                    ["door", "n03222176"]
                    ])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES)
num_classes = len(classes)

# Placeholders
x = tf.placeholder(tf.float32, shape=[input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, 3])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

x_reshaped = tf.reshape(x, [-1, input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, 3])

# First convolutional layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# First fully-connected layer
W_fc1 = weight_variable([56 * 56 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 56 * 56 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second fully-connected layer
W_fc2 = weight_variable([1024, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    image_batch, label_batch = train_dataset.next_batch(50)
    if i % 5 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: image_batch, y_: label_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: test_dataset.images, y_: test_dataset.labels, keep_prob: 1.0}))


def main():
    ids = classes[:, 1]
    input_data.download_dataset(ids, NUM_IMAGES)


if __name__ == "__main__":
    main()
