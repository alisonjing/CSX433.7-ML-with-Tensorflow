#!/usr/bin/env python
#1. Include a section line with your name:

# Name: Alison Jing Huang -- jing01ucsb@gmail.com
# ID: X111012
# Machine Learning with Tensorflow HW #2
# Due 6/10/2018

# Recreate the graph and visualize it in Tensorboard using:
#1. Placeholder for an input array with dtype float32 and shape None
#2. Scopes for the input, middle section and final node f
#3. Feed the placeholder with an array A consisting of 100 normally distributed
#random numbers with Mean = 1 and Standard deviation = 2
#4. Save your graph and show it in Tensorboard
#5. Plot your input array on a separate figure
#6. Make sure you comment your code well and provide your name on top of your work
#7. Email yout Github link(or code directly) to me including yout .py file+screenshots of Tensorboard.

# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

#Fixing random state of reproducibility
np.random.seed(3)

# Constructing the "input_placeholder" section
# Create a placehlder vector of length 3 with data type float32
with tf.name_scope("input_placeholder"):
    a = tf.placeholder(tf.float32, shape=None, name ="Input_a")

# Constructing the "Middle_section"
# Use the placeholder as it were any other Tensor object
with tf.name_scope("Middle_section"):
    b = tf.reduce_prod(a, name="b")
    c = tf.reduce_mean(a, name ="c")
    d = tf.reduce_sum(a, name ="d")
    e = tf.add(b,c, name ="e")

# Constucting the "final_node"
with tf.name_scope("Final_node"):
    f = tf.multiply(e,d, name="f")

# Run the graph session
with tf.Session() as sess:

# Feed the placeholder with an array A consisting of 100
# normally distributed random numbers with Mean =1 and standard deviation
# = 2

# Create an array of 100 normally distributed
    normal_dist = np.random.normal(1.0, 2.0, 100)

# Final node with the array of 100 normally distributed random numbers
# with Mean =1 and standard deviation  = 2
    print("f = %s" % sess.run(f, feed_dict={a:normal_dist }))

    sess.graph.as_graph_def()
    file_writer = tf.summary.FileWriter('./HW2_graph', sess.graph)


# Plotting the normally distributed randomly generated numbers
#plt.interactive(False)
plt.hist(normal_dist, bins= 20)
plt.title("Histogram of Normal Distribution of Randomly Generated values")
plt.xlabel('Values')
plt.ylabel('Frequency')
#plt.ioff()
plt.show()

#Plotting the normally distributed generated numbers by the array index
#plt.interactive(False)
N = 100
color = np.random.rand(N)
plt.title('Normal Distribution of Randomly Generated Values by Index')
plt.scatter(list(range(0,100)), normal_dist, c=color, alpha = 0.5)
plt.xlabel('Index_a')
plt.ylabel('a')
plt.show()
