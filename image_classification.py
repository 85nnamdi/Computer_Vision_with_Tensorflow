import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
#from skimage import color
import math

#read the files
#path = '.\positives\ '
p_path = 'path\\positives\\'
n_path = 'path\\negatives\\'

def read_files(dir, extension):
    files = [f for f in glob.glob(dir + "**/*." + extension, recursive=True)]
    return files

def im_to_feature(im_path):
    files = read_files(im_path,"png")
    b_avg =[]
    g_avg = []
    r_avg =[]
    feat =[]
    r_varance = []
    g_varance = []
    b_varance = []
    sobel =[]
    for f in files:
        im = plt.imread(f)
        r_avg.append(np.max(im[:,:,0]))
        g_avg.append(np.max(im[:,:,1]))
        b_avg.append(np.max(im[:,:,2]))
        r_varance.append(np.var(im[:,:,0]))
        g_varance.append(np.var(im[:,:,1]))
        b_varance.append(np.var(im[:,:,2]))
    feat = np.array((r_avg, g_avg, b_avg,r_varance,g_varance,b_varance))

    return np.transpose(feat)

#convert to color
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def multiVarGaus(mu, sigma, x):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

#GDA function
def gda(p_features, n_features, x): #pass in the target image
    #target = np.sum(n_image)/n_image.count() #ratio of examples =0.5
    mu0 = np.mean(n_features,axis=0) #negative images
    mu1 = np.mean(p_features,axis=0) #positive images

    p_sigma = np.matrix(np.cov(np.transpose(p_features)))

    n_sigma = np.matrix(np.cov(np.transpose(n_features)))

    result0 = multiVarGaus(mu0,n_sigma,x)

    result1 = multiVarGaus(mu1, p_sigma, x)

    return (result1-result0)

#definition for the positives
p_features = im_to_feature(p_path)
#definition for the Negatives
n_features = im_to_feature(n_path)

#compute sobel filter

p=0
n=0
for i in range(30):
    result = gda(p_features,n_features,p_features[i])
    if result >0:
        p+=1
    result = gda(p_features, n_features, n_features[i])
    if result < 0:
        n += 1
print("Positives ",format(p))
print('Negatives  ',format(n))
print('percentage correct',format((n+p)/60))
#plt.imshow(mag.astype('double'), cmap='gray')
#plt.show()

#GDA
