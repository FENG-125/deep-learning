
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
# 加载
import LeNet5_inference

BATCH_SIZE = 100  # 一次训练的数据个数
LEARNING_RATE_BASE = 0.01  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数中的系数
TRAINING_STEPS = 6000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODE_SAVE_PATH = "A:/Project/my_mnist/model/LeNet5"
MODE_NAME = "model.ckpt"


def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet5_inference.IMAGE_SIZE,
                                        LeNet5_inference.IMAGE_SIZE,
                                        LeNet5_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_inference.inference(x,False ,regularizer)
    #训练轮数的变量设为 不可训练
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('moving_average'):
        #初始化滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        #计算所有可训练变量的影子变量
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):
        #argmax()得到正确答案数字
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.argmax(y_, 1))
        #交叉熵的平均值(损失)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #总损失                                    权值正则化损失和
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope('train_step'):
        #设置指数衰减的学习率
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
        #优化算法优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                               global_step=global_step)
        #更新参数和参数的滑动平均值
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # 初始化tf持久化类
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                reshaped_xs = np.reshape(xs,[BATCH_SIZE, LeNet5_inference.IMAGE_SIZE,
                                        LeNet5_inference.IMAGE_SIZE,
                                        LeNet5_inference.NUM_CHANNELS])
                _, loss_value, step = sess.run([train_op, loss, global_step],
                feed_dict = {x: reshaped_xs, y_: ys})

                if i % 1000 == 0:
                    print("After %d training step , loss on training batch is %g" % (step, loss_value))
                    #saver.save(sess, os.path.join(MODE_SAVE_PATH, MODE_NAME),
                     #          global_step=global_step)

    #summary_writer = tf.summary.FileWriter('A:/Project/my_mnist/log/', sess.graph)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()