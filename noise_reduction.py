# -*- coding:utf-8 -*-

# Noise reduction
# using NN

import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np

# 問題設定
noise_mu = 15
noise_sigma = 10
size1 = 20
numberOfPoints = 2 * size1 + 1
numberOfData = 100
numberOfTest = 3
numberOfTrain = numberOfData - numberOfTest

# アルゴリズム設定
# Number of perceptrons at 2nd layer
# default: 10000
numberOf2ndPerceptrons = 10000
#
stddevOfPerceptrons = 0.03  # 0.1 #0.03
learningRate = 0.9  # 0.1 #0.9


def Gaussian(x, mu, sigma):
    return math.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


samplePoints = range(-size1, size1 + 1)
signal = np.zeros((numberOfData, numberOfPoints))
print(len(samplePoints))
obs = np.zeros((numberOfData, numberOfPoints))

dSigs = np.random.rand(numberOfData).astype("float32")

for j in range(numberOfData):
    for i in range(numberOfPoints):
        signal[j][i] = Gaussian(samplePoints[i], 0, 0.5 + 0.05 * size1 * dSigs[j])
        obs[j][i] = signal[j][i] + 0.2 * Gaussian(
            samplePoints[i], noise_mu, noise_sigma)
        # print(str(i) + " " + str(samplePoints[i]) + " " + str(obs[i]))

# Visualize for check
for j in range(min(3, numberOfData)):
    plt.scatter(samplePoints, signal[j])
    plt.scatter(samplePoints, obs[j])
    plt.show()

# 入力データを定義
x = tf.placeholder(tf.float32, [None, numberOfPoints])

# 入力層から中間層
w_1 = tf.Variable(tf.truncated_normal(
    [numberOfPoints, numberOf2ndPerceptrons], stddev=stddevOfPerceptrons), name="w1")
b_1 = tf.Variable(tf.zeros([numberOf2ndPerceptrons]), name="b1")
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

# 中間層から出力層
w_2 = tf.Variable(tf.truncated_normal(
    [numberOf2ndPerceptrons, numberOfPoints], stddev=stddevOfPerceptrons), name="w2")
b_2 = tf.Variable(tf.zeros([numberOfPoints]), name="b2")
# out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)
out = tf.matmul(h_1, w_2) + b_2

# 誤差関数
y = tf.placeholder(tf.float32, [None, numberOfPoints])
loss = tf.reduce_mean(tf.square(y - out))

# 訓練
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# 評価
# correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 初期化
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

#         sess.run(train_step, feed_dict={x:obs[j] ,y:signal[j]})
    sess.run(
        train_step, feed_dict={x:obs[0:numberOfTrain], y:signal[0:numberOfTrain]})

    outVal = sess.run(out, feed_dict={x:obs[numberOfTrain:numberOfData]})
    for i in range(len(outVal)):
        print(str(i) + str(len(outVal[i])))
        plt.scatter(samplePoints, outVal[i], s=numberOfPoints)
        plt.scatter(samplePoints, signal[numberOfTrain + i])
        plt.show()
#    print(outVal)

#         if step % 10 == 0:
#             acc_val = sess.run(accuracy ,feed_dict={x:test_images, y:test_labels})
#             print('Step %d: accuracy = %.2f' % (step, acc_val))
