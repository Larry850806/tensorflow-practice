import tensorflow as tf
import numpy as np

# generate 100 float from 0 to 1
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 4.5

### create tensorflow structure START
# Weights is a one dimension array, random range from -1 to 1
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

# learning rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)

# train is to minimize loss
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure END

sess = tf.Session()
sess.run(init)

for step in range(201):
    if(step % 20 == 0):
        print(step, sess.run(Weights), sess.run(biases))
    sess.run(train)

# Ref: https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/
