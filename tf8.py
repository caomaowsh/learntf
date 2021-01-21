"""
tensorflow: 1.12.0
python: 3.6.9
"""
import tensorflow as tf 
import numpy as np 

## 
np_x = np.random.uniform(-1,1, (1000,1))
np_y = np.power(np_x, 2) + np.random.normal(0,0.1, size=npx.shape)

np_x_train, np_x_test = np.split(np_x, [800])
np_y_train, np_y_test = np.split(np_y, [800])

tf_x = tf.placeholder(np_x_train.dtype, np_x_train.shape)
tf_y = tf.placeholder(np_y_train.dtype, np_y_train.shape)

## 
dataset = tf.data.Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.repeat(3)
iterator = dataset.make_initializable_iterator()

##  ?????
bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, out)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

##
sess = tf.Session()
## initialize iterator
init_v = tf.global_variables_initializer()
init_i = iterator.initializer
sess.run([init_i, init_v], feed_dict={tfx: np_x_train, tfy: np_y_train})

for step in range(201):
	try:
		## train
		_, train_loss = sess.run([train_op, loss])
		if step % 10 == 0:
			## test
			test_loss = sess.run(loss, {bx: npx_test, by: npy_test})
			print('step: {}'.format{step}, 
				'|train_loss: {}'.format{train_loss}, '|test_loss: {}'.format{test_loss})
	# if training takes more than 3 epochs, training will be stopped
	except tf.errors.OutOfRangeError:
		print('Finish the last epoch.')
		break