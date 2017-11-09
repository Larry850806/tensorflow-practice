import tensorflow as tf
import numpy as np

attributes = [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
]

labels = [False, True, True, False]

data = np.array(attributes, np.bool_)
target = np.array(labels, np.bool_)

feature_columns = [tf.contrib.layers.real_valued_column("")]

learning_rate = 0.1
epoch = 1e4

estimator = tf.contrib.learn.SKCompat(
    tf.contrib.learn.DNNClassifier(
        hidden_units=[3],
        feature_columns=feature_columns,
        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
        activation_fn=tf.nn.sigmoid
    )
)

estimator.fit(x=data, y=target, steps=epoch)
predictions = estimator.predict(x=data)["classes"]

for i in range(len(predictions)):
    prediction = bool(predictions[i])
    print("{} xor -> actual {}, predict {}".format(data[i], target[i], prediction))
