"""
tensorflow: 1.12.0
python: 3.6.9
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as numpy

tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal()

def save():
	print('This is save')
	#
	tf_x = tf.placeholder(tf.float32, x.shape)
	tf_y = tf.placeholder(tf.float32, y.shape)
	l = tf.layers.dense(tf_x, 10, tf.nn._relu)
	o = tf.layers.dense(l, 1)
	loss = tf.losses.mean_square_error(tf_y, o)
	optimizer = tf.train.GradientDecsentOptimizer(learning_rate=0.5)
	train_op = optimizer.minimize(loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		for step in range(100):
			sess.run(train_op, feed_dict={tf_x:x, tf_y:y})

		# meta_graph is not recommended ?
		saver.save(sess, './params', write_meta_graph=False)

		pred, l = sess.run([o, loss], feed_dict={tf_x:x, tf_y:y})

		plt.figure(1, figsize=(10,5))
		plt.subplot(121)
		plt.scatter(x, y)
		plt.plot(x, pred, 'r-', lw=5)
		plt.text(-1,1.2, 'save loss=%.4f' % l, font_dict={'size': 15, 'color': 'red'})


def reload():
	print('This is reload')
	## first need to buold entire net again and then resore
	tf_x = tf.placeholder(tf.float32, x.shape)
	tf_y = tf.placeholder(tf.float32, y.shape)
	l = tf.layers.dense(tf_x, 10, tf.nn._relu)
	o = tf.layers.dense(l, 1)
	loss = tf.losses.mean_square_error(tf_y, o)

	## dont need to initialize variables, just restoring trained variables
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, './params')
		plt.subplot(122)
		plt.scatter(x, y)
		plt.plot(x, pred, 'r-', lw=5)
		plt.text(-1,1.2, 'reload loss=%.4f' % l, font_dict={'size': 15, 'color': 'red'})
		plt.show()

save()

##  destroy previous net
tf.reset_default_graph()

reload()
