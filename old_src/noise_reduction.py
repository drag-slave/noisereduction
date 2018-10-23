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
miniBatchSize = 20

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
print("# of sample points: ", len(samplePoints))
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
x = tf.placeholder(tf.float32, [None, numberOfPoints], name="x_input")

# 入力画像をログに出力
img = tf.reshape(x, [-1, numberOfPoints, 1, 1])
tf.summary.image("log_input_data", img, 2)

# 入力層から中間層
with tf.name_scope("second_layer"):
    w_1 = tf.Variable(tf.truncated_normal(
        [numberOfPoints, numberOf2ndPerceptrons], stddev=stddevOfPerceptrons), name="w1")
    b_1 = tf.Variable(tf.zeros([numberOf2ndPerceptrons]), name="b1")
    h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

    # 中間層の重みの分布をログ出力
    tf.summary.histogram('log_w_1', w_1)

# 中間層から出力層
with tf.name_scope("output_layer"):
    w_2 = tf.Variable(tf.truncated_normal(
        [numberOf2ndPerceptrons, numberOfPoints], stddev=stddevOfPerceptrons), name="w2")
    b_2 = tf.Variable(tf.zeros([numberOfPoints]), name="b2")
    # out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)
    out = tf.matmul(h_1, w_2) + b_2

# 誤差関数
y = tf.placeholder(tf.float32, [None, numberOfPoints], name="y_input")
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - out))

    # 誤差をログ出力
    tf.summary.scalar("log_loss", loss)

# 訓練
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# 初期化
init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter("logs", sess.graph)

    sess.run(init)

#         sess.run(train_step, feed_dict={x:obs[j] ,y:signal[j]})
    for j in range(0, numberOfTrain, miniBatchSize):
        idxEnd = min(j + miniBatchSize, numberOfTrain)
        print("Index from ", j, " to ", idxEnd - 1)

        sess.run(
            train_step, feed_dict={x:obs[j:idxEnd], y:signal[j:idxEnd]})

        test_data = obs[numberOfTrain:numberOfData]

        # tf.reshapeの第1引数に、numpy.arrayを入れていいのかあやしい。
        test_images = np.reshape(test_data, [
            numberOfData - numberOfTrain, numberOfPoints])  # , 1, 1])
        test_labels = signal[numberOfTrain:numberOfData]

        # test_dataからノイズ除去
        outVal = sess.run(out, feed_dict={x:test_data})

        if j % 1 == 0:
            for i in range(len(outVal)):
                print(str(i) + str(len(outVal[i])))
                plt.scatter(samplePoints, outVal[i], s=numberOfPoints)
                plt.scatter(samplePoints, signal[numberOfTrain + i])
                # plt.show()

            # ログを取る処理を実行する（出力はログ情報が書かれたプロトコルバッファ）
            # test_labelsを2次元arrayで与えていいか不明。
            summary_str = sess.run(summary_op, feed_dict={x:test_images, y:test_labels})
            # ログ情報のプロトコルバッファを書き込む
            summary_writer.add_summary(summary_str, idxEnd)
