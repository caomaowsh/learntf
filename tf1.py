'''
Python = 3.6.9
tensorflow = 1.12.0
'''

import tensorflow as tf

## --------------- 常量tf.constant()----------------
m1 = tf.constant([[2,2]])
m2 = tf.constant(2, shape=[2,1])

dot_operation = tf.matmul(m1, m2)
print(dot_operation)

## ------------ 会话tf.Session() 方式1 ------------
sess = tf.Session()
result = sess.run(dot_operation)
print(result)
sess.close()
## ------------ 会话tf.Session() 方式2 ------------
with tf.Session() as sess:
	result_ = sess.run(dot_operation)
	print(result_)

## ------------ 占位符tf.placeholder() ------------
x1 = tf.placeholder(dtype=tf.float32, shape=None)
x2 = tf.placeholder(dtype=tf.float32, shape=None)
z = x1+x2

y1 = tf.placeholder(dtype=tf.float32, shape=[2,1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1,2])
w = tf.matmul(y1, y2)

with tf.Session() as ses:
	###  依次运行
	zz = sess.run(z, feed_dict={x1:1, x2:2})
	ww = sess.run(w, feed_dict={y1:[[2],[2]], y2:[[3,3]]})
	print(zz)
	print(ww)
	###  同时运行
	zz_, ww_ = sess.run(
		[z, w],
		feed_dict={x1:1, x2:2, y1:[[2],[2]], y2:[[3,3]]})
	print(zz_)
	print(ww_)

## ------------ 变量tf.Variable() ------------
var = tf.Variable(0, name='Counter')#初始值，变量名
print(var.name)

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for _ in range(3):
		sess.run(update_operation)
		print(sess.run(var))