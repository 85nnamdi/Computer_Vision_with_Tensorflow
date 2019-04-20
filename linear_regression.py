import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#setup computational graph
x_data = np.load('linreg_x.npy')
y_data = np.load('linreg_y.npy')

x = tf.placeholder(tf.float32, shape=x_data.shape,name='X')
y= tf.placeholder(tf.float32, shape=y_data.shape,name='Y')



print(y_data.shape)

w = tf.Variable(np.random.normal(0,1),name='W')
b = tf.Variable(np.random.normal(0,1),name='B')

y_predicted = w*x+b


#set up a training operation
loss_function = tf.losses.mean_squared_error(y_data,y_predicted)
grad_decent_optimizer =tf.train.GradientDescentOptimizer(0.1)
minimize_op = grad_decent_optimizer.minimize(loss_function)

l_summary = tf.summary.scalar(name='loss',tensor=loss_function)
w_summary =tf.summary.scalar(name='W value',tensor=w)
b_summary =tf.summary.scalar(name='B Value',tensor=b)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    for k in range(100):
        _,l,wk,bk = sess.run([minimize_op,loss_function,w,b],feed_dict={x:x_data, y:y_data})
        print('Iteration {:d}: Loss{:f}, w ={:f}, b= {:f}'.format(k,l,wk,bk))
        ls, ws, bs = sess.run([l_summary, w_summary, b_summary], {x:x_data, y:y_data})

        summary_writer.add_summary(ls, global_step=k)  # writes loss summary
        summary_writer.add_summary(ws, global_step=k)  # writes summary for variable w
        summary_writer.add_summary(bs, global_step=k)  # writes summary for variable b

#Start plotting the line
xs = np. linspace ( -1.0 , 1.0 , num =20)
ys = wk*xs+bk

plt.plot(x_data,y_data,'bo')
plt. plot (xs ,ys ,'g')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
