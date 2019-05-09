import numpy as np
import matplotlib.pyplot as plt
import math

#initialize data randomly
x = np.random.random(size=(100,1))#generate_random(0,1,100)
epsilon = np.random.uniform(-0.3,0.3,size =(100,1))
y = np.sin(2*np.pi*x)+epsilon
learning_rate = 0.1


#initialize the parameter values to random number between -0.5 and 0.5
theta0 = np.random.uniform(-0.5,0.5,1)
theta1= np.random.uniform(-0.5,0.5,1)
theta3= np.random.uniform(-0.5,0.5,1)
theta2= np.random.uniform(-0.5,0.5,1)

x_plot = np.linspace(0,1,100,endpoint=True)
hypothesis = np.empty([100,1]) #update vector or the hypothesis
hyp_ori_plot = (theta0)+(theta1*x_plot)+(theta2*pow(x_plot,2))+(theta3*pow(x_plot,3))
err = []
#print(h_update.shape)

#plot original model
plt.plot(x_plot,np.transpose(hyp_ori_plot), label="Initial model")
plt.scatter(x,y, label="Datapoint")



for itr in np.arange(5000):
    for i in np.arange(0, np.size(x)):
        #hypothesis
        hypothesis[i] = (theta0)+(theta1*x[i])+(theta2*pow(x[i],2))+(theta3*pow(x[i],3))
        #np.append(h_update, hypothesis)

        #print(hypothesis[i])
        #recalculte theta
        theta0 = theta0+learning_rate*(y[i]-hypothesis[i])
        theta1 = theta1+learning_rate*(y[i]-hypothesis[i])*x[i]
        theta2 = theta2 + learning_rate * (y[i] - hypothesis[i] )* pow(x[i],2)
        theta3 = theta3 + learning_rate * (y[i] - hypothesis[i] )* pow(x[i],3)


    # calaculate the error
    err_value = np.square((hypothesis[i] - y[i]))  #
    err.append(err_value)

    hyp_plot = (theta0) + (theta1 * x_plot) + (theta2 * pow(x_plot, 2)) + (theta3 * pow(x_plot, 3))
    y_plot = np.sin(2 * np.pi * x_plot)
    plt.plot(x_plot,np.transpose(hyp_plot), label="Learned function")
    plt.pause(0.01)

plt.plot(x_plot,y_plot,label="Original Signal")

plt.legend()
plt.show()

#plot error
plt.plot(range(len(err)),err)
plt.show()
#plt.show()
