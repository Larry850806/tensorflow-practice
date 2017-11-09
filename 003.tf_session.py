import tensorflow as tf

matrix1 = tf.constant([[1, 2]])
matrix2 = tf.constant([[3],[4]])

# 1 2 x 3 = 1 x 3 + 2 x 4 = 11
#       4

print(matrix1)

# matrix multiply
# np.dot(matrix1, matrix2)
product = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
    # auto close
