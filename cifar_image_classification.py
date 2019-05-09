import numpy as np
import os
import tensorflow as tf

"""
unpickle function from CIFAR-10 website
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    return d

def create_dataset_from_files(files):
	rawdata = []
	labels = []
	for f in files:
		d = unpickle(f)
		rawdata.extend(d["data"])
		labels.extend(d["labels"])

	rawdata = np.array(rawdata)

	red = rawdata[:,:1024].reshape((-1,32,32))
	green = rawdata[:,1024:2048].reshape((-1,32,32))
	blue = rawdata[:,2048:].reshape((-1,32,32))

	data = np.stack((red,green,blue), axis=3)
	labels = np.array(labels)

	return data, labels

cifar_data_path = ".\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\" # replace with your path
cifar_training_files = [os.path.join(cifar_data_path, 'data_batch_{:d}'.format(i)) for i in range(1,6)]
cifar_testing_files = [os.path.join(cifar_data_path, 'test_batch')]

train_data, train_labels = create_dataset_from_files(cifar_training_files)
test_data, test_labels = create_dataset_from_files(cifar_testing_files)

print(train_data.shape, train_data.dtype) # (50000, 32, 32, 3) -- 50000  images, each 32 x 32 with 3 color channels (UINT8)
print(train_labels.shape, train_labels.dtype) # 50000 labels, one for each image (INT64)

print(test_data.shape, test_data.dtype) # 10000 images, each 32 x 32 with 3 color channels (UINT8)
print(train_labels.shape, train_labels.dtype) # 10000 labels, one for each image (INT64)


image  = tf.placeholder(tf.uint8, shape=[None, 32, 32, 3],name="Image")
labels = tf.placeholder(tf.int64, shape=[None,],name="labels")

pre_image= tf.image.convert_image_dtype(image, dtype=tf.float32)

  # Convolutional Layer #1
conv1_1 = tf.layers.conv2d(inputs=pre_image,filters=32,kernel_size=[3,3], activation=tf.nn.relu)
conv1_2= tf.layers.conv2d(inputs=conv1_1,filters=32,kernel_size=[3,3], activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2,2],strides=2)

  # Convolutional Layer #2
conv2_1 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=[3,3], activation=tf.nn.relu)
conv2_2= tf.layers.conv2d(inputs=conv2_1,filters=32,kernel_size=[3,3], activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2,2],strides=2)

#flatten the previous layers
pool2_fattened = tf.layers.flatten(pool2,name="flatten")

#dense layer fully connected layers
fcl = tf.layers.dense(pool2_fattened, units=10, activation=None)

#define a loss function
loss_function = tf.losses.sparse_softmax_cross_entropy(labels,logits=fcl)
grad_decent_optimizer =tf.train.GradientDescentOptimizer(0.1)
minimize_op = grad_decent_optimizer.minimize(loss_function)

#calulate the accuracy
predictions = tf.argmax(fcl , axis =1)
correct_preds = tf.equal ( labels , predictions )
accuracy = tf.reduce_mean (tf.cast(correct_preds, tf.float32 ) )

#Monitor loss and acurracy
l_summary = tf.summary.scalar(name='loss summary',tensor=loss_function)
a_summary =tf.summary.scalar(name='accuracy summary',tensor=accuracy)

batchsize = 32
# start the training
with tf.Session() as sess:
    #write to log file
    summary_writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    #draw sample indeces
    idx = np.random.choice(train_data.shape[0], batchsize, replace=False)
    for batch in range(batchsize):
        _,l = sess.run([minimize_op,loss_function],feed_dict={image:train_data, labels:train_labels})
        #print('Iteration {:d}: Loss{:f}, w ={:f}, b= {:f}'.format(k,l,wk,bk))
        ls, a_s = sess.run([l_summary, a_summary], {image:x_data, labels:y_data})

        summary_writer.add_summary(ls, global_step=k)  # writes loss summary
        summary_writer.add_summary(a_s, global_step=k)  # writes accuracy summary

        if batch % 100 == 0:
            print('Batch {:d} done'.format(batch))


    test_loss , test_accuracy = sess.run([ loss , accuracy ], feed_dict ={images : test_data , labels : test_labels })
    print ('Test loss : {:f} -- test accuracy : {:f}'. format ( test_loss, test_accuracy ) )

# Automate the logviewer from within python
subprocess.Popen(["tensorboard", "--logdir=./logs"])
webbrowser.open('http://localhost:6006', new=1)
