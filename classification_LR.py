import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.txt")

# sclice the first row column of the data
x = data[0:100, 0]

# slice the second column of the data
y = data[:, 1]
labels = data[:, 2]
alpha = 0.1
epoch = 50

# plot x and y
plt.scatter(x, y, c=labels, s=50)

# theta with 3 parameters
theta = np.random.uniform(-0.01, 0.01, 3)

x_plot = np.linspace(-3, 3, 100)

# initial plot line
model_x2 = (theta[0] + theta[1] * x_plot) / (-theta[2])

# plot the initial model
plt.plot(x_plot, model_x2)

# stochastic gradient descent
# theta_j = theta_j+alpha*(labels(i)-g)

z = []

# Parameter should be x and y
model_param = data[:, 0:2]

#error derivative
err = []
# parameters
z=np.empty([100,1]) #htpothesis with empty content
g=np.empty([100,1]) #sigmoid function



# hypothesis

for iteration in range(100):
    for i in range(np.size(x)):
        #model_x = theta[0]+(alpha.dot())
        z[i] = theta[0]+theta[1]*x[i]+theta[2]*y[i]
        g[i]= 1/(1+np.exp(-z[i]))
        theta[0] = theta[0]+(alpha*(labels[i]-g[i]))
        theta[1] = theta[1] + (alpha * (labels[i] - g[i])) * x[i]
        theta[2] = theta[2] + (alpha * (labels[i] - g[i])) * y[i]

    # calaculate the error
    err_value = np.sum(np.square((labels[i]-g[i])))  #
    err.append(err_value)

# learning
# theta+alpha(labels(i) - g)
#hyp_plot_z = theta[0]+theta[1]*x_plot+theta[2]*x_plot
hyp_plot_z= (theta[0] + theta[1] * x_plot) / (-theta[2])
hyp_plot = 1/(1+np.exp(np.transpose(-hyp_plot_z)*x))
#y_plot = np.sin(2 * np.pi * x_plot)
plt.plot(x_plot,hyp_plot_z, label="Learned function")


#plt.plot(x_plot,y_plot,label="Original Signal")

plt.legend()
plt.show()

#plot error
plt.plot(range(len(err)),err)
plt.show()
#plt.show()

plt.show()
