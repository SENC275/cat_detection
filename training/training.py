import glob
import os

import cv2

import numpy as np
import tensorflow as tf


l2_weight = 0.0001
momentum = 0.9
initial_lr = 0.1
lr_step_epoch = 100.0
lr_deacy = 0.1


def parse_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    image = image/255.0
    return image


cat_files = glob.glob("./cat/*")
cat_labels = [1 for item in cat_files]
dog_files = glob.glob("./dog/*")
dog_labels = [0 for item in dog_files]

train_cats = cat_files[0:int(len(cat_files)*0.5)]
train_cats_labels = cat_labels[0:int(len(cat_files)*0.5)]

test_cats = cat_files[int(len(cat_files)*0.9):]
test_cats_labels = cat_labels[int(len(cat_files)*0.9):]

train_dogs = dog_files[0:int(len(dog_files)*0.5)]
train_dogs_labels = dog_labels[0:int(len(dog_files)*0.5)]

test_dogs = dog_files[int(len(dog_files)*0.9):]
test_dogs_labels = dog_labels[int(len(dog_files)*0.9):]

total_files = cat_files + dog_files
total_labels = cat_labels + dog_labels
assert len(total_files) == len(total_labels), "files and labels are not equal"

train_files = train_cats + train_dogs
train_labels = train_cats_labels + train_dogs_labels

test_files = test_cats + test_dogs
test_labels = test_cats_labels + test_dogs_labels

train_file_tensor = tf.convert_to_tensor(train_files, dtype=tf.string)
train_label_tensor = tf.convert_to_tensor(train_labels, dtype=tf.int64)

test_file_tensor = tf.convert_to_tensor(test_files, dtype=tf.string)
test_label_tensor = tf.convert_to_tensor(test_labels, dtype=tf.int64)

train_dataset = tf.data.Dataset.from_tensor_slices((train_file_tensor,
                                                    train_label_tensor))
train_dataset = train_dataset.shuffle(buffer_size=20000)
train_dataset = train_dataset.batch(128)

test_dataset = tf.data.Dataset.from_tensor_slices((test_file_tensor,
                                                   test_label_tensor))
test_dataset = test_dataset.batch(128)


class Model:
    def __init__(self, sess):
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'
        self._build_net()

    def _relu(self, x, leakness=0.0, name=None):
        if leakness > 0.0:
            name = 'lrelu' if name is None else name
            return tf.maximum(x, x*leakness, name='lrelu')
        else:
            name = 'relu' if name is None else name
            return tf.nn.relu(x, name='relu')

    def _conv(self, x, filter_size, out_channel, strides, pad='SAME', name='conv'):
        in_shape = x.get_shape()
        with tf.variable_scope(name):
            kernel = tf.get_variable('kernel',
                                     [filter_size, filter_size, in_shape[3], out_channel],
                                     tf.float32,
                                     initializer=tf.random_normal_initializer(
                                        stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)
                                     ))
            if kernel not in tf.get_collection(self.WEIGHT_DECAY_KEY):
                tf.add_to_collection(self.WEIGHT_DECAY_KEY, kernel)
            conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
        return conv

    def _fc(self, x, out_dim, name='fc'):
        with tf.variable_scope(name):
            w = tf.get_variable('weights',
                                [x.get_shape()[1], out_dim],
                                tf.float32,
                                initializer=tf.random_normal_initializer(
                                    stddev=np.sqrt(1.0/out_dim)
                                ))
            if w not in tf.get_collection(self.WEIGHT_DECAY_KEY):
                tf.add_to_collection(self.WEIGHT_DECAY_KEY, w)
            b = tf.get_variable('biases',
                                [out_dim],
                                tf.float32,
                                initializer=tf.constant_initializer(0.0))
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
        return fc

    def _bn(self, x, is_train, global_step=None, name='bn'):
        moving_average_decay = 0.9
        with tf.variable_scope(name):
            decay = moving_average_decay
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            mu = tf.get_variable('mu',
                                  batch_mean.get_shape(),
                                  tf.float32,
                                  initializer=tf.zeros_initializer,
                                  trainable=False)
            sigma = tf.get_variable('sigma',
                                    batch_var.get_shape(),
                                    tf.float32,
                                    initializer=tf.ones_initializer,
                                    trainable=False)
            beta = tf.get_variable('beta',
                                   batch_mean.get_shape(),
                                   tf.float32,
                                   initialzer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma',
                                     batch_var.get_shape(),
                                     tf.float32,
                                     initialzer=tf.ones_initializer)
            update = 1.0 - decay
            update_mu = mu.assign_sub(update*(mu-batch_mean))
            update_sigma = simga.assign_sub(update*(sigma-batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

            mean, var = tf.cond(is_train, lambda: (batch_mean. batch_var),
                                lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean,  var, beta, gamma, 1e-5)
        return bn

    def _build_net(self):
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, 128, 128, 3], name='input_image')
        self.Y = tf.placeholder(tf.int32, [None,])
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)

        filters = [16, 32, 64]

        print("building----net ")
        x = self._conv(self.X, 3, 16, 1, name='init_conv')

        for i in range(3):
            with tf.variable_scope("block_%d" % i) as scope:
                print("buildibg -------{}".format(scope.name))
                shortcut = x
                shortcut = tf.nn.max_pool(x,
                                          [1,2,2,1],
                                          [1,2,2,1],
                                          'VALID')

                x = self._conv(x, 3, filters[i], 1, name='conv1')
                x = self._relu(x, name='relu1')
                x = self._conv(x, 3, filters[i], 1, name='conv2')
                x = self._relu(x, name='relu2')
                x = self._conv(x, 3, filters[i], 1, name='conv3')
                x = self._relu(x, name='relu3')
                x = tf.nn.max_pool(x,
                                   [1,2,2,1],
                                   [1,2,2,1],
                                   'VALID')

                x = x + shortcut
                if i < 2:
                    x = self._conv(x, 3, filters[i+1], 1, name='conv4')
                print("block---", i)
                print("shape---", x.get_shape())


        with tf.variable_scope('last_unit') as scope:
            print("building-------{}".format(scope.name))
            x = tf.reshape(x, [-1, 16*16*64])
            x = self._fc(x, 2)

        self.logits = tf.identity(x, name="logits")

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y
        ))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(self.cost)
        correct_prediction = tf.equal(
            tf.cast(tf.argmax(self.logits, 1), tf.int32), self.Y
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def predict(self, x_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test,
                                        self.keep_prob: 1})
    def get_accuracy(self, x_test, y_test, traninig=False):
        return self.sess.run([self.cost, self.accuracy],
                             feed_dict={
                                self.X: x_test,
                                self.Y: y_test,
                                self.keep_prob: 1
                             })
    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.accuracy],
                             feed_dict={
                                self.X: x_data,
                                self.Y: y_data,
                                self.training: training,
                                self.keep_prob: 0.5
                             })

train_iterator = train_dataset.make_initializable_iterator()
train_one_batch = train_iterator.get_next()

test_iterator = test_dataset.make_initializable_iterator()
test_one_batch = test_iterator.get_next()


with tf.Session() as sess:
    model = Model(sess)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(300):
        print("epoch-------------------", epoch)
        train_loss, train_acc, n_batch = 0, 0, 0
        sess.run(train_iterator.initializer)
        try:
            while True:
                paths, label = sess.run(train_one_batch)
                paths = [path.decode('utf-8') for path in paths]
                images = [parse_image(path) for path in paths]
                cost, _, accuracy = model.train(images, label)
                print("train cost------", cost)
                train_loss += cost
                train_acc += accuracy
                n_batch += 1
        except tf.errors.OutOfRangeError:
            pass

        val_loss, val_acc, n_test_batch = 0, 0, 0
        sess.run(test_iterator.initializer)
        try:
            while True:
                paths, test_label = sess.run(test_one_batch)
                paths = [path.decode('utf-8') for path in paths]
                test_images = [parse_image(path) for path in paths]
                val_cost, val_accuracy = model.get_accuracy(test_images, test_label)
                val_loss += val_cost
                val_acc += val_accuracy
                n_test_batch += 1
        except tf.errors.OutOfRangeError:
            pass

        if val_acc / n_test_batch > 0.55:
            model_name = "epoch_{}_acc_{}".format(epoch, val_acc / n_test_batch)
            os.mkdir("./cat_models/epoch_{}_{}".format(epoch, val_acc / n_test_batch))
            saver.save(sess, os.path.join("./cat_models/epoch_{}_{}".format(epoch, val_acc / n_test_batch), model_name + ".ckpt"))

        print("epoch num ------", epoch)
        print("-----------------------")
        print("    trian loss: %f" % (np.sum(train_loss)/n_batch))
        print("    train acc: %f" % (np.sum(train_acc)/n_batch))
        print("    val loss: %f" % (np.sum(val_loss)/n_test_batch))
        print("    val acc %f" % (np.sum(val_acc)/n_test_batch))
