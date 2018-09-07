#-*- coding:utf-8 -*-

# Noise reduction
# using NN

import matplotlib.pyplot as plt
import tensorflow as tf

# Define a single scalar Normal distribution.
dist = tf.distributions.Normal(loc=0., scale=1.)

# Get 10 samples, returning a 10 tensor.
distSamples = dist.sample([1000])

#初期化
init = tf.global_variables_initializer()

with tf.Session() as sess:
 
    sess.run(init)    
#     sess.run(distSamples)
    evaluatedSamples = distSamples.eval()
#     print("samples = " + str(evaluatedSamples))
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(evaluatedSamples, bins=11)
ax.set_title('Gaussian')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
    

