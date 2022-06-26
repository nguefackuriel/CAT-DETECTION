#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Neural Network and Tensorflow
# 
# The goal of this work is to create a neural network with Tensorflow to recognize if an image is a cat picture or not
# 
# 
# Let's Start!!
# 
# **What you will learn:**
# - Construct the general architecture of a learning algorithm which contains:
#      - Parameters initialization
#      - Compute cost function and its gradient
#      - Use of an optimization algorithm(Gradient descent)
# - Put together the three last functions in a main function.
# - Test the model with your own picture.

# ## 1 - Packages ##
# 
# First, let's execute this cell to import all packages that we will need for this task.
# - [numpy](www.numpy.org) is the fundamental package for scientific computing with python.
# - [h5py](http://www.h5py.org) is the package used to interact with a set of data stored in an H5 file.
# - [matplotlib](http://matplotlib.org) is the famous library to plot in python.
# - [PIL](http://www.pythonware.com/products/pil/) et [scipy](https://www.scipy.org/) are used import your image and test with the model built.

# - Install tensorflow version 1.15 with the command line: pip install tensorflow==1.15
# 
# - Install open cv version 3.4.2 with the command line:pip install opencv-python==3.4.2.17, when you are done execute also pip install opencv-contrib-python==3.4.2.17 

# In[98]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from math import *
import scipy


import tensorflow as tf
from tensorflow.python.framework import ops

from PIL import Image
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# ## 2 - Overview of the problem ##
# 
# **Problem** : You have a set of data ("data.h5") containing :
#      - A training set of images m_train labeled as cat (y=1) or non-cat (y=0)
#      - A test set of images m_test labeled as cat or not-cat
#      - Each image has the shape (num_px, num_px, 3) where 3 is for the 3 channels (RVB). So, each image is a square (height = num_px) et (width = num_px).
# 
# You will create an algorithm to recognize simple image that can classify images as cat or non-cat images.
# 
# Let's explore the dataset. Load the datset by executing the following code.

# In[99]:


# Load data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# We have added "_orig" at the end of images dataset (train and test) because we will preprocess them. After the preprocessing, we will have now train_set_x et test_set_x (labels sets train_set_y and test_set_y don't need preprocessing).
# 
# Each line of train_set_x_orig and test_set_x_orig is a table representing an image. You can visualize an example by executing the following code. You can also modify the value `index` and reexecute to see other images.

# In[100]:


# Example of image

index = 5
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


# 
# 
# Many software bugs in deep learning come from having matrix/vector dimensions that do not match. If you can keep your matrix/vector dimensions straight, you will go a long way to eliminating many bugs.
# 
# **Exercice :** Find values for :
#      - m_train (number of training examples)
#      - m_test (number of test examples)
#      - num_px (= height = width of a training image)
# Don't forget that `train_set_x_orig` is a numpy array in a shape of (m_train, num_px, num_px, 3). For example, you can access to `m_train` by writing `train_set_x_orig.shape[0]`.

# In[ ]:



m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[2]


print ("Number of training set: m_train = " + str(m_train))
print ("Number of test set: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image has a shape: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape : " + str(train_set_x_orig.shape))
print ("train_set_y shape : " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape : " + str(test_set_y.shape))


# For more commodity, you have now to resize images of shape (num_px, num_px, 3) to a numpy array of shape (num_px $*$ num_px $*$ 3, 1). After that, our training dataset (and test) is a numpy array where each column represent a flatten image. It should be m_train column (respectively m_test).
# 
# **Exercice :** Reshape the training and test datasets so that the images of size (num_px, num_px, 3) are flattened into single shape vectors (num\_px $*$ num\_px $*$ 3, 1).
# 
# A trick to flatten a matrix X of form (a,b,c,d) into a matrix X_flatten of form (b$*$c$*$d, a) is to use :
# ```python
# X_flatten = X.reshape(X.shape[0], -1).T # X.T is the transpose of X
# ```
# 
# In addition to this, you will convert each label into a single vector, as shown in Figure 1. Run the cell below to do this.

# In[102]:


# Flatten the training and test images
X_train_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
X_test_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(train_set_y, 2)
Y_test = convert_to_one_hot(test_set_y, 2)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# **Note** that 12288 comes form $64 \times 64 \times 3$. Each image is square, 64 by 64 pixels, and 3 is for RGB colors. Please make sure that all these shapes make sense to you before continuing.
# 
# To represent color images, the red, green, and blue (RGB) channels must be specified for each pixel, so the pixel value is actually a vector of three numbers ranging from 0 to 255.
# 
# A common preprocessing step in machine learning is to center and normalize your dataset, which means you subtract the mean of the entire numpy array from each example, and then divide each example by the standard deviation of the entire numpy array. But for image data sets, it is simpler and more convenient and works almost as well to simply divide each row of the data set by 255 (the maximum value of a pixel channel).
# 
# <!-- When training your model, you will multiply the weights and add biases to some of the initial inputs in order to observe neuron activations. Then, you perform backpropagation with the gradients to train the model. But, it is extremely important that each feature has a similar range so that our gradients do not explode. You will see this in more detail later in the lectures.
# 
# Let's standardize our dataset.
# 

# In[103]:


X_train = X_train_flatten/255.
X_test = X_test_flatten/255.


# <font color='bleu'>
# **What you need to remember:**
# 
# The common steps in preprocessing a new dataset are:
# - Determine the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# - Reshape the datasets so that each example is now a vector of size (num_px \* num_px \* 3, 1)
# - "Standardize" the data

# ### 2.1 - Creating placeholders
# 
# Your first task is to create placeholders for "X" and "Y". This will allow you to transmit your training data later when you run your session.
# 
# **Exercise:** implement the function below to create placeholders in tensorflow.

# In[104]:


#  FONCTION: create_placeholders

def create_placeholders(n_x, n_y):
    """
    Create placeholders for tensorflow session.
    
    Arguments:
     n_x -- scalar, shape of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
     n_y -- scalar, number of classes (from 0 to 1, so -> 2)
    
     Return:
     X -- space reserved for data entry, form [n_x, None] and type "tf.float32"
     Y -- space reserved for input labels, of form [n_y, None] and type "tf.float32"
    
     Advices:
     - You will use None because because it allows us to be flexible on the number of examples you will use for the placeholders.
        In fact, the number of examples during the test/train is different.
     """

   
    X = tf.placeholder(tf.float32, shape = [n_x, None], name = "X")
    Y = tf.placeholder(tf.float32, shape = [n_y, None], name = "Y")
   
    
    return X, Y


# In[105]:


X, Y = create_placeholders(12288, 2)
print ("X = " + str(X))
print ("Y = " + str(Y))


# In[106]:


import tensorflow
print(tensorflow.__version__)


# In[107]:


# Change x value in the feed_dict
sess = tf.Session()
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()


# ## 3 - General architecture of the learning algorithm
# 
# It's time to design a simple algorithm to distinguish cat images from non-cat images.
# 
# You will build a logistic regression, using a neural network mindset. The following figure explains why **Logistic regression is actually a very simple neural network!
# 
# <img src="images/LogReg_kiank.png" style="width:650px;height:400px;">
# 
# **Mathematical expression of the algorithm** :
# 
# For an example $x^{(i)}$:
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$
# 
# The cost is then compute by suming all training examples :
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$
# 
# **Keys steps** :
# In this work, we will construct these different steps: 
#      - Initialize model parameters
#      - Learn model parameters by minimizing cost
#      - Use parametrs learn to make predictions (on the test set)
#      - Analyse results and conclude

# ## 4 - Build the different parts of our algorithm ##
# 
# The different steps to build neural network are:
# 1. Define model structure (such as the number of input entities)
# 2. Initialize model parameters 
# 3. Loop :
#      - Compute the current cost (forward propagation)
#      - Compute the current gradient (backward propagation)
#      - Update parameters (gradient descent)
#  
# We will build 1-3 separately and integrate them into a function we will call `model()`.
# 
# ### 4.1 - Sigmoide computing
# 
# Tensorflow offers a variety of neural networks functions often used, such as « tf.sigmoid » and « tf.softmax ». For this code, let's compute the sigmoid function for one input. Pour cet exercice, calculons la fonction sigmoïde d'une entrée.
# 
# We will do this code using a placeholder variable "x". When we run the session, we must use the flow dictionary to pass the "z" input. In this code, we will (i) create a placeholder "x", (ii) define the operations needed to calculate the sigmoid using "tf.sigmoid", and then (iii) run the session.
# 
# ** Exercice ** : Implement the sigmoid function below. We must use the following elements :
# 
# - `tf.placeholder(tf.float32, name = "...")`
# - `tf.sigmoïd(...)`
# - `sess.run(..., feed_dict = {x:z})`
# 
# 
# Note that it exists two typical ways to create and use sessions in Tensorflow:
# 
# **Method 1 :**
# ```python
# sess = tf.Session()
# # Execute variables initialization (if necessary), execute the operations
# result = sess.run(..., feed_dict = {...})
# sess.close() # Close the session
# ```
# **Method 2:**
# ```python
# with tf.Session() as session :
#     # launch variables initialization (if needed), launch operations
#     result = sess.run(..., feed_dict = {...})
#     # It helps you close the session :)
# ```

# In[108]:


# FONCTION: sigmoid

def sigmoid(z):
    """
    Compute sigmoid of z

    Arguments:
    z -- A scalar or an numpy arrayof any size.

    Return:
    s -- sigmoid(z)
    """

   
     # Create a placeholder for x. Name it "x".
    x = tf.placeholder(tf.float32, name = "x")

     # Compute sigmoid(x)
    y = tf.sigmoid(x)

     # Create a session and execute it. Let's use method 2 explained above.
     # We must use a feed_dict to pass the value of z to x.
    with tf.Session() as sess:
         # Exécutez la session et appelez la sortie "résultat"
        result = sess.run(y, feed_dict = {x:z})

    
    
    return result


# In[109]:


print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))


# 
# ### 4.2 - Parameters initialization
# 
# Your second task consist to initialize parameters in Tensorflow.
# 
# **Exercice :** Implement the below function to initialize the parameters in Tensorflow. We will use Xavier initialization for weights and zero initialization for bias. Shapes are given below. For example for W1 and b1, we will use: 
# 
# ```python
# W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
# b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
# ```
# Let's use `seed = 1` to make sure your results are the same as mine.

# In[110]:


# FONCTION: initialize_parameters

def initialize_parameters():
    """
     Initialize parameters to build a neural network with Tensorflow. Shape are :
                         W1 : [25, 12288]
                         b1 : [25, 1]
                         W2 : [12, 25]
                         b2 : [12, 1]
                         W3 : [2, 12]
                         b3 : [2, 1]
    
     Return:
     parameters -- A dictionnary of tensors containing W1, b1, W2, b2, W3, b3
     """
    
    tf1.set_random_seed(1)                   # For your random number to correspond to mine
        
    
    W1 = tf1.get_variable('W1', [25,12288], initializer = tf2.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf1.get_variable('b1', [25,1], initializer = tf1.zeros_initializer())
    W2 = tf1.get_variable('W2', [12,25], initializer = tf2.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf1.get_variable('b2', [12,1], initializer = tf1.zeros_initializer())
    W3 = tf1.get_variable('W3', [2,12], initializer = tf2.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf1.get_variable('b3', [2,1], initializer = tf1.zeros_initializer())
  

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# In[111]:


tf1.reset_default_graph()
with tf1.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# ### 4.3 - Forward propagation with Tensorflow
# 
# We are going now to implement the forward propagation module in tensorflow. The function will take a dictionary of parameters and terminate the forward pass. The functions we will use are:
# 
# - `tf.add(...,...)` to add
# - `tf.matmul(...,...)` to perform array multiplication
# - `tf.nn.relu(...)` to apply ReLU activation
# 
# It is important to note that the forward propagation stops at "z3". The reason for this is that in tensorflow, the last linear layer output is given as input to the function calculating the loss. Therefore, you do not need "a3"!
# 

# In[112]:


## FONCTION: forward_propagation

def forward_propagation(X, parameters):
    """
     Implement the forward propagation for the model : LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
     Arguments:
     X -- placeholder for the input data set, form (input size, number of examples)
     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3
                   shapes are given in initialize_parameters

     Return:
     Z3 -- the output of the last unit LINEAR
     """
    
    # Retrieve parameters from the "parameters" dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
                                                                              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                               # Z3 = np.dot(W3, A2) + b3
   
    
    return Z3


# In[113]:


tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 2)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))


# ### 4.4 Compute the cost
# 
# As seen previously, it is very easy to compute the cost by using :
# ```python
# tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))
# 
# - It is important to know that the `logits` and `labels` entries in `tf.nn.softmax_cross_entropy_with_logits` are supposed to be of form (number of examples, num_classes). So we have transposed Z3 and Y for you.
# - Also, `tf.reduce_mean` essentially does the summation over the examples.

# In[114]:


# FONCTION: compute_cost 

def compute_cost(Z3, Y):
    """
     Compute the cost
    
     Arguments:
     Z3 -- output of the forward propoagation (output of the last unit LINEAR), of shape (6, number of samples)
     Y -- "true" placeholder vector labels, same shape as Z3
    
     Return:
     cost - cost function Tensor 
     """
    
   # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
   
    
    return cost


# In[115]:


tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))


# ### 2.5 - Backward Propagation and parameters update
# 
# This is where we become grateful to programming frameworks. All backpropagation and parameter updating is taken care of in 1 line of code. It is very easy to integrate this line into the model.
# 
# After calculating the cost function. We will create an object "optimizer". We need to call this object with the cost when running tf.session. When called, it will perform an optimization on the given cost with the chosen method and learning rate.
# 
# For example, for gradient descent, the optimizer would be:
# ``python
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# ```
# 
# To do the optimization, you would do:
# ``python
# _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
# ```
# 
# This computes the backpropagation through the tensorflow graph in reverse order. From cost to inputs.
# 
# **Note** When coding, we often use `_` as a "throwaway" variable to store values that we won't need to use later. Here, `_` takes the evaluated value of `optimizer`, which we don't need (and `c` takes the value of the variable `cost`).
# 
# 

# ### 2.6 - Build the model
# 
# Now we'll put it all together!
# 
# 

# In[116]:


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # training characteristics
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # training labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test characteristics
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test labels

    classes = np.array(test_dataset["list_classes"][:]) # class list
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training samples
    mini_batches = []
    np.random.seed(seed)
    
    # Etape 1: Mélanger (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Etape 2: Partition (shuffled_X, shuffled_Y). Minus the final case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of minibatch of size mini_batch_size in our partitioning
  
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    #  Management of the final case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve parameters from the "parameters" dictionary 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3
    


# In[118]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # be able to rerun the model without overwriting the tf variables
    tf.set_random_seed(1)                             # to keep the results consistent
    seed = 3                                          # to keep the results consistent
    (n_x, m) = X_train.shape                          # (n_x: input shape, m : number of training samples)
    n_y = Y_train.shape[0]                            # n_y : output shape
    costs = []                                        # To store the cost
    
    # Create placeholder of shape (n_x, n_y)
    
    X, Y = create_placeholders(n_x, n_y)
   
    # Initialize parameters
    
    parameters = initialize_parameters()
   
    
    # Forward propagation:Build the forward propagation
  
    Z3 = forward_propagation(X, parameters)
  
    
    # Cost function: Add the cost function 
   
    cost = compute_cost(Z3, Y)
  
    
    # Backward Propagation: Define the optimizer function of tensorflow. Use AdamOptimizer.
  
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
  
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the tensorflow session
    with tf.Session() as sess:
        
        # Execute the initialization
        sess.run(init)
        
        # start the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Define a cost linked to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatch of sizee minibatch_size 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
               
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X :minibatch_X, Y: minibatch_Y})
               
                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost at each epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Save parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Compute the good prediction
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Compute the accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


# In[119]:


import math
parameters = model(X_train, Y_train, X_test, Y_test)


# **Comment**: The training accuracy is close to 100%. This is a good integrity check: the model works and has a high enough capacity to fit the training data. The accuracy of the test is 76%. This is actually not bad for this simple model, given the small data set we used and the fact that logistic regression is a linear classifier. 
# 
# 

# ### IMPORT LIBRAIRIES

# In[120]:


import imageio
import cv2


# In[121]:


cv2.__version__


# ### TEST WITH IMAGES DOWNLOAD ON INTERNET
# 
# ## 3 - Test with your own image  ##
# 
# You can use your own image and see the result of your template. To do this:
#      1. Click on "File" in the top bar of this notebook, then click on "Open" to access your image folder.
#      2. Add your image to this Jupyter Notebook's directory, in the "images" folder.
#      3. Change the name of your image in the following code
#      4. Run the code and check if the algorithm is correct (1 = cat, 0 = non-cat)!

# In[122]:




my_image = "la_defense.jpg"


# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(imageio.imread(fname))
image = image/255.
my_image = cv2.resize(image, (64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


# ### TEST WITH IMAGES FROM THE TEST SET

# In[123]:


# Example of a picture that was wrongly classified.
index=4
plt.imshow(X_test[:,index].reshape((num_px, num_px, 3)))
image =X_test[:,index].reshape((num_px, num_px, 3))/255.
image = cv2.resize(image, (64,64)).reshape((1, 64*64*3)).T
print ("y = " + str(Y_test[1,index]) + ", Your algorithm predicts: y = " + str(np.squeeze(predict(image, parameters))))


# ## In this part, we will test our model using a graphical interface created with Tkinter

# **1) Importing libraries (to import a library you write import "library name" as "diminutive"; to import a package specific to a library, you write from "library name" import "package name")
# - Import the tkinter library as tk : tkinter is one of the python libraries to design graphical interfaces.
# Example: import tkinter as tk
# - Import the filedialog package from the tkinter library
# - Import all packages from tkinter with the * element
# - Import the ImageTk and Image packages from the PIL library

# In[ ]:


import tkinter as tk                            # Import tkinter
from tkinter import filedialog                            # Import of the filedialog package from the tkinter library
from tkinter import *                             # Import of all tkinter packages
from PIL import ImageTk, Image                             # Import of ImageTk and Image packages from the PIL library
import numpy                             # Import numpy


# ### Execute this cell to create a dictionnary

# In[87]:


#dictionnary to label the classes of our images
classes = { 0:"Ce n'est pas un chat",
            1:"C'est un chat" 
 }


# ### Execute this cell to initialize Tkinter

# In[88]:


#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Classification des images')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)


# ### Execute the cell below to create the function classify that we will use to make prediction

# In[89]:


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((64, 64))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image.reshape((1, 64*64*3)).T
    pred = int(np.squeeze(predict(image, parameters)))
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign) 


# ### The following function create a button "Classify the image" that will call our function "classify" created above

# In[90]:


def show_classify_button(file_path):
    classify_b=Button(top,text="Classifier the Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)


# ### To access the directory of your computer to choose one of the images you have download online, we need to create an upload_image function. Run the following code to do this
#  

# In[91]:


def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


# ### Execute the following code to have access to the graphic interface user and do your test.

# In[ ]:


upload=Button(top,text="Charger une image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Connaitre la natuure de l'image",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

