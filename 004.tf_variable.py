import tensorflow as tf

state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

# new_value = state + 1
# state = new_value
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# ref: https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-4-variable/