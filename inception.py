import input_data
from input_data import create_datasets
import numpy as np
import tensorflow as tf

NUM_IMAGES = 100

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


def conv2d(x, W, strides):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES)
num_classes = len(classes)

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, input_data.IMAGE_WIDTH * input_data.IMAGE_HEIGHT, 3])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

x_reshaped = tf.reshape(x, [-1, input_data.IMAGE_WIDTH, input_data.IMAGE_HEIGHT, 3])

# First convolutional layer, (224, 224, 3) to (56, 56, 48)
W_conv1 = weight_variable([11, 11, 3, 48])
b_conv1 = bias_variable([48])

h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1, [1, 4, 4, 1]) + b_conv1)

# Second convolutional layer, (56, 56, 48) to (28, 28, 128)
W_conv2 = weight_variable([5, 5, 48, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1, 1, 1, 1]) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Third convolutional layer, (28, 28, 128) to (14, 14, 192)
W_conv3 = weight_variable([3, 3, 128, 192])
b_conv3 = bias_variable([192])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, [1, 1, 1, 1]) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# Fourth convolutional layer, (14, 14, 192) to (14, 14, 192)
W_conv4 = weight_variable([3, 3, 192, 192])
b_conv4 = bias_variable([192])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, [1, 1, 1, 1]) + b_conv4)

# Fifth convolutional layer, (14, 14, 192) to (14, 14, 128)
W_conv5 = weight_variable([3, 3, 192, 128])
b_conv5 = bias_variable([128])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, [1, 1, 1, 1]) + b_conv5)

# First fully-connected layer
W_fc1 = weight_variable([14 * 14 * 128, 512])
b_fc1 = bias_variable([512])

h_conv5_flat = tf.reshape(h_conv5, [-1, 14 * 14 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second fully-connected layer
W_fc2 = weight_variable([512, 512])
b_fc2 = bias_variable([512])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Third fully-connected layer
W_fc3 = weight_variable([512, num_classes])
b_fc3 = bias_variable([num_classes])

y_score = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
y_logit = tf.nn.softmax(y_score)

# Training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logit, y_))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100000):
    image_batch, label_batch = train_dataset.next_batch(50)
    sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
    if i % 5 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        train_cost = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g, cost %g" % (i, train_accuracy, train_cost))

    if i % 50 == 0:
        y_scores = sess.run(y_score, feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        print "Scores: ", y_scores[0]
        y_logits = sess.run(y_logit, feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
        print "Logits: ", y_logits[0]

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: test_dataset.images, y_: test_dataset.labels, keep_prob: 1.0}))


def main():
    ids = classes[:2, 1]
    input_data.download_dataset(ids, NUM_IMAGES)


if __name__ == "__main__":
    main()
