#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:30:29 2018

@author: daniellay1

"""

import numpy as np
import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt

t0 = time.time()
os.chdir(os.getcwd()+'/Configs')

np.random.seed(1)

"""
Reads in data used.
"""
def read_configs(input_file):
    with open(input_file) as file:
        plaqs = file.readlines()

    temp_configs = np.asarray(plaqs)
    configs = []
    for s in range(0,len(temp_configs)):
        config = temp_configs[s].split(" ")
        temp_config = np.zeros(len(config))
        for i in range(0,len(config)-1):
            temp_config[i] = float(config[i])
        configs.append(temp_config)
    configs = np.asarray(configs)
    plaqs = configs[:,0:-1]
    return plaqs

def read_acts(input_file):
    with open(input_file) as file:
        actions = file.readlines()
    temp_actions = np.asarray(actions)
    actions = []
    for s in range(0,len(temp_actions)):
        action = temp_actions[s].split("\n")
        action = np.asarray([float(action[0])])
        actions.append(action)
    actions = np.reshape(np.asarray(actions),(-1,))
    return actions

#configs = read_configs('4x4 Trans Configs.txt')
actions = read_acts('4x4 Actions.txt')

#Expands the actions so that translated configurations have the same action.
expand_acts = np.zeros(configs.shape[0])
for i in range(0,actions.shape[0]):
    expand_acts[16*i:16*(i+1)] = actions[i]

actions = np.expand_dims(expand_acts,axis=-1)
del expand_acts

"""
Defining neural net.
"""
#def loss_std(y,y_):
#    tf.add(y,-y_)


#Layer sizes
in_size = configs.shape[1] #Number of columns of data
h1_size = 64
h2_size = 8

#Placeholder variables
X = tf.placeholder(tf.float32,shape=(None,in_size))
Y = tf.placeholder(tf.float32,shape=(None,1))

#Layer 1
w1 = tf.Variable(tf.random_normal([h1_size,in_size]))
b1 = tf.Variable(tf.random_normal(shape=(h1_size,1)))
z1 = tf.nn.relu(tf.add(tf.matmul(w1,tf.transpose(X)),b1))
z1 = tf.nn.dropout(z1,keep_prob=0.5)

#Layer 2
w2 = tf.Variable(tf.random_normal([h2_size,h1_size]))
b2 = tf.Variable(tf.random_normal([h2_size,1]))
z2 = tf.nn.relu(tf.add(tf.matmul(w2,z1),b2))
z2 = tf.nn.dropout(z2,keep_prob=0.5)

#Output layer
wo = tf.Variable(tf.random_normal([1,h2_size]))
bo = tf.Variable(tf.random_normal([1,1]))
zo = tf.nn.relu(tf.add(tf.matmul(wo,z2),bo)) #prediction for action

#Sets up cost function and tells it to use Adam optimizer
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y,
                                                   predictions=zo))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)#stands for 'train_operation'

#Checks accuracy of the net.
#pred = tf.nn.relu(zo)
acc = tf.reduce_mean(tf.metrics.mean_squared_error(labels = Y, predictions = zo))

"""
Running neural net.
"""
#Parameters for training
n_epochs = 2000
n_keep = 50
acc_vals = np.zeros(n_keep)
batch_size = 5000

#Splits into train/test sets
train_configs = configs[0:int(0.5*configs.shape[0]),:]
val_configs = configs[int(0.9*configs.shape[0]):int(0.91*configs.shape[0]),:]

train_actions = actions[0:int(0.9*configs.shape[0])]
val_actions = actions[int(0.9*configs.shape[0]):int(0.91*configs.shape[0])]

# Initialize the variables (i.e. assign their default value)
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

#Actually trains the net.
def train(write=True):
    if write == True:
        run_time = time.ctime()
        os.chdir(os.getcwd()+'/Data')
        open('Weight vals '+str(run_time)+'.txt','w').close()
        open('Bias vals '+str(run_time)+'.txt','w').close()
        open('Cost vals '+str(run_time)+'.txt','w').close()
    
    with tf.Session() as sess:
        tf.set_random_seed(1)
        
        m = 0
        sess.run(init_g)
        sess.run(init_l)
        
        for i in range(n_epochs):
            perm = np.random.permutation(train_configs.shape[0])
            batch_perm = perm[0:batch_size]
            
            batch_configs = train_configs[batch_perm,:]
            batch_actions = train_actions[batch_perm,:]
            
            sess.run(train_op,feed_dict={X: batch_configs, Y: batch_actions})
                
            if i % int(n_epochs/n_keep) == 0:
                acc_vals[m] = sess.run(acc, feed_dict = {X: val_configs, Y: val_actions})
                t_ep = time.time()
                if write == True:
                    w1_temp, w2_temp, wo_temp = sess.run([w1,w2,wo])
                    w_file = open('Weight vals '+str(run_time)+'.txt','a')
                    w_file.write(str(i))
                    for w in w1_temp.reshape(-1):
                        w_file.write(' '+str(w))
                    for w in w2_temp.reshape(-1):
                        w_file.write(' '+str(w))
                    for w in wo_temp.reshape(-1):
                        w_file.write(' '+str(w))
                    w_file.write('\n')
                    w_file.close()
                    
                    b1_temp, b2_temp, bo_temp = sess.run([b1,b2,bo])
                    b_file = open('Bias vals '+str(run_time)+'.txt','a')
                    b_file.write(str(i))
                    for b in b1_temp.reshape(-1):
                        b_file.write(' '+str(b))
                    for b in b2_temp.reshape(-1):
                        b_file.write(' '+str(b))
                    for b in bo_temp.reshape(-1):
                        b_file.write(' '+str(b))
                    b_file.write('\n')
                    b_file.close()
                    
                    cost_file = open('Cost vals '+str(run_time)+'.txt','a')
                    cost_file.write(str(i)+' '+str(acc_vals[m])+'\n')
                    cost_file.close()
                    
                print('Finished epoch '+str(i)+' in time '+str(t_ep-t0))
                m += 1
        ep_vals = np.arange(0,n_epochs,int(n_epochs/n_keep))
        plt.plot(ep_vals,acc_vals,'.-')
            #print(sess.run([zo]))
train()


t1 = time.time()
print('Took '+str(t1-t0)+' seconds.')