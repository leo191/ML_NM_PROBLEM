import tensorflow as tf


W = tf.Variable([0.3],tf.float32)
b = tf.Variable([-0.3],tf.float32)

x = tf.placeholder(tf.float32)
lin_mod = W*x + b
y = tf.placeholder(tf.float32)

squared_delta = tf.square(lin_mod - y )

loss = tf.reduce_sum(squared_delta)
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)




init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    print(sess.run([W,b]))
