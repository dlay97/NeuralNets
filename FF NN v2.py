#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:48:48 2018

@author: daniellay1

With blocks this time!

FINITE DIFFERENCE TESTING STILL FAILS TO FOLLOW GRADIENT
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt

from block_param_funcs import *

t_start = time.time()

os.chdir(os.getcwd()+'/Plaqs')

np.random.seed(5)

"""
Reads in data used.
"""
def read_file(input_file):
    with open(input_file+' Plaquettes.txt') as file:
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

    with open(input_file+" Actions.txt") as file:
        actions = file.readlines()
    temp_actions = np.asarray(actions)
    actions = []
    for s in range(0,len(temp_actions)):
        action = temp_actions[s].split("\n")
        action = np.asarray([float(action[0])])
        actions.append(action)
    actions = np.reshape(np.asarray(actions),(-1,))
    return plaqs, actions

"""
Stuff for neural net construction. May need to wrap a command that generates
all layers later.
"""
# weight_dims should be an array with the output sizes 
def full_layer_params(out_size,in_size,param_type='Rand',loc=0,scale=1):
    if param_type == 'Zeros':
        weights = np.zeros((out_size,in_size))
        biases = np.zeros((out_size))
    elif param_type == 'Rand':
        weights = np.random.normal(loc,scale,size=(out_size,in_size))
        biases = np.random.normal(loc,scale,size=(out_size))
    elif param_type == 'Ones':
        weights = scale*np.ones((out_size,in_size))
        biases = scale*np.ones((out_size))
    else:
        print('Pick a layer type.')
        return
    
    return biases, weights
    
"""
Computes stuff to show how the training works, e.g. the cost function.
"""
def cost(real_actions,nn_actions):
    n_actions = len(real_actions)
    cost_out = 0
    for t in range(0,n_actions):
        cost_out += 1/n_actions*(nn_actions[t]-real_actions[t])**2
    return cost_out

#Returns the cost function evaluated at a single configuration
def d_cost(real_actions,nn_actions):
    cost_out = 2*(nn_actions - real_actions)
    return cost_out

"""
Stuff for running neural net - may move around later.
"""
def activate(array,func='Relu'):
    if func == 'Sigmoid':
        activ_out = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            if array[i] >= 30:
                # Checks for overflow warning; computer can't evaluate e^(large number)
                # so use asymptotic form.
                activ_out[i] = 1
            else:
                activ_out[i] = np.exp(array[i])/(np.exp(array[i])+1)
    elif func == 'Relu':
        activ_out = array.copy()
        activ_out[activ_out<0] = 0
    return activ_out

def d_activ(array,func='Relu'):
    if func == 'Sigmoid':
        activ = activate(array,'Sigmoid')
        activ_out = np.multiply(activ,np.ones(array.shape[0])-activ)
    elif func == 'Relu':
        activ_out=array.copy()
        activ_out[activ_out>=0]=1
        activ_out[activ_out<0]=0 
    return activ_out

def feed_through_layer(config,weight_in,bias_in,latt_dim,n_plaqs,n_loops,n_blocks,
                       activ='True'):
    if type(weight_in) == tuple:
        #Imported from block_param_funcs
        weight_thru = full_mat(unwrap_weights(weight_in,latt_dim,n_plaqs,n_loops
                                              ,n_blocks))
        print('Unwrapping weights')
    else:
        weight_thru = weight_in
    if len(weight_thru.shape) == 2:
        weight_thru = weight_thru.reshape((1,-1,n_blocks))
        print('Reshaping weights')
    n_params = weight_thru.shape[0]
    
    z_out = np.zeros((n_params,n_blocks))
    a_out = z_out.copy()
    for k in range(0,n_blocks):
        z_out[:,k] = np.dot(weight_thru[:,:,k],config).reshape(-1,) + bias_in[:,k]
        if activ == 'True':
            a_out[:,k] = activate(z_out[:,k])
        else:
            a_out[:,k] = z_out[:,k]
    return z_out, a_out

# Returns the a and z tensors constructed by feeding through the plaquettes
def feed_through_all(configs,weights,biases,latt_dim,n_plaqs,n_loops,n_blocks,verbose=False):
    feed_time_0 = time.time()
    z_out = []
    a_out = []
    
    #shape not inferred from weights because weights must be unpacked
    z1 = np.zeros((configs.shape[0],configs.shape[1],n_blocks)) #len of config, n_blocks, n_configs
    a1 = z1.copy()
    for t in range(0,len(configs)):
        config_in = configs[t,:].reshape((-1,1))
        z1[t,:,:], a1[t,:,:] = feed_through_layer(config_in,weights[0],biases[0],\
                      latt_dim,n_plaqs,n_loops,n_blocks)
    z_out.append(z1)
    a_out.append(a1)
    #shape inferred from weights
    z2 = np.zeros(configs.shape[0])
    
    for t in range(0,len(configs)):
        
        config_in = a1[t,:,:].reshape((-1,1))
        z2[t] = np.dot(weights[1],config_in) + biases[1]
    a2 = activate(z2)#because actions are nonnegative, activation is fine here
    z_out.append(z2)
    a_out.append(a2)
    feed_time_1 = time.time()
    if verbose == True:
        print('Feeding took '+str(feed_time_1-feed_time_0)+' seconds.')
    return z_out, a_out

"""
If error exists in the W0 gradient term, it's in the chain_rule function or in the
feed_all function. Both of those appear in both places.
"""
"""
Finite difference gradient terms.
"""
#Finite difference gradient.
def fd_w0(configs,actions,weights,biases):
    eps = 0.001
    dw_out = np.zeros(weights[0].shape)
    for i in range(0,n_params):
        for j in range(0,n_params):
            for k in range(0,n_blocks):
                w0_temp = np.zeros(weights[0].shape)
                w0_temp[i,j,k] = eps
                w0_temp = chain_rule(w0_temp,latt_dim,n_plaqs,n_blocks,n_loops)
                w0_plus = weights[0].copy()
                w0_minus = weights[0].copy()
                w0_plus += w0_temp
                w0_minus -= w0_temp
                w_plus = [w0_plus,weights[1]]
                w_minus = [w0_minus,weights[1]]
                z_p, a_p = feed_through_all(configs,w_plus,biases,latt_dim,
                                                  n_plaqs,n_loops,n_blocks)
                z_m, a_m = feed_through_all(configs,w_minus,biases,latt_dim,
                                                  n_plaqs,n_loops,n_blocks)
                cost_p = cost(actions,a_p[1])
                cost_m = cost(actions,a_m[1])
                dw_out[i,j,k] = (cost_p - cost_m)/(2*eps)
    return dw_out

def fd_b0(configs,actions,weights,biases):
    eps = 0.001
    db_out = np.zeros(biases[0].shape)
    for i in range(0,n_params):
        for k in range(0,n_blocks):
            b0_p = biases[0].copy()
            b0_p[i,k] += eps
            b0_m = biases[0].copy()
            b0_m[i,k] -= eps
            b_plus = [b0_p,biases[1]]
            b_minus = [b0_m,biases[1]]
            z_p,a_p=feed_through_all(configs,weights,b_plus,latt_dim,n_plaqs,n_loops,
                                     n_blocks)
            z_m,a_m=feed_through_all(configs,weights,b_minus,latt_dim,n_plaqs,n_loops,
                                     n_blocks)
            cost_p = cost(actions,a_p[1])
            cost_m = cost(actions,a_m[1])
            db_out[i,k] = (cost_p - cost_m)/(2*eps)
    return db_out

def fd_w1(configs,actions,weights,biases):
    eps = 0.00001
    dw_out = np.zeros(weights[1].shape)
    for i in range(0,np.prod(weights[1].shape)):
        w_p = weights[1].copy()
        w_m = w_p.copy()
        w_p[0,i] += eps
        w_m[0,i] -= eps
        w_p = [weights[0],w_p]
        w_m = [weights[0],w_m]
        z_p, a_p = feed_through_all(configs,w_p,biases,latt_dim,
                                                  n_plaqs,n_loops,n_blocks)
        z_m, a_m = feed_through_all(configs,w_m,biases,latt_dim,
                                                  n_plaqs,n_loops,n_blocks)
        cost_p = cost(actions,a_p[1])
        cost_m = cost(actions,a_m[1])
        dw_out[0,i] = (cost_p-cost_m)/(2*eps)
    return dw_out
    
def fd_b1(configs,actions,weights,biases):
    eps = 0.001
    b_p = biases[1] + eps
    b_m = biases[1] - eps
    b_p = [biases[0],b_p]
    b_m = [biases[0],b_m]
    z_p, a_p = feed_through_all(configs,weights,b_p,latt_dim,
                                                  n_plaqs,n_loops,n_blocks)
    z_m, a_m = feed_through_all(configs,weights,b_m,latt_dim,
                                                  n_plaqs,n_loops,n_blocks)
    cost_p = cost(actions,a_p[1])
    cost_m = cost(actions,a_m[1])
    db_out = (cost_p - cost_m)/(2*eps)
    return db_out

def fd_grad(configs,actions,weights,biases):
    db0 = fd_b0(configs,actions,weights,biases)
    db1 = fd_b1(configs,actions,weights,biases)
    dw0 = fd_w0(configs,actions,weights,biases)
    dw1 = fd_w1(configs,actions,weights,biases)
    
    w_out = [dw0,dw1]
    b_out = [db0,db1]
    return w_out, b_out

#Computes the gradient for all of the configs (FOR 2 LAYERS ONLY).
def compute_grad(configs,actions,weights,biases,latt_dim,n_plaqs,n_loops,n_blocks,
                 debug=False,verbose=False):
    if debug == True:
        verbose = True
    n_layers_in = len(weights)
    if n_layers_in != n_layers:
        print('Check the number of layers!')
        return
    z_init, a_init = feed_through_all(configs,weights,biases,latt_dim,n_plaqs,
                                      n_loops,n_blocks,verbose)
    deltas = []
    #d_activ returns 0's and 1's, because Relu(z)={0,x<0;x,x>0}.
    d1 = d_cost(actions,a_init[-1])* d_activ(z_init[-1])
    deltas.append(d1)
    
    d0 = np.zeros(shape=z_init[0].shape)
    for i in range(0,len(configs)):
        dR = d_activ(z_init[0][i,:,:]).reshape((-1,1))
        weightIn = np.transpose(weights[1] * d1[i])
        d0_temp = np.multiply(weightIn,dR)
        d0[i,:,:] = d0_temp.reshape((-1,n_blocks))
        #Returns a \delta value for each parameter shaped to fit the block structure.
        #Reshapes correctly according to the unravelling above.
    deltas.append(d0)
    
    #Reverses the order of the list to match that weights[0] are the weights attached
    #to the 1st layer (0th is input; has nothing attached)- now, deltas[0] is the 
    #change applied to the weights at the 1st layer
    deltas = list(reversed(deltas))
    
    n_params = configs.shape[1]
    n_configs = configs.shape[0]
    dW0 = np.zeros(shape=weights[0].shape)
    db0 = np.zeros(shape=biases[0].shape)
    dW1 = np.zeros(shape=weights[1].shape)
    db1 = np.zeros(shape=biases[1].shape)
    for t in range(0,n_configs):
        for i in range(0,n_params):
            for j in range(0,n_params):
                for k in range(0,n_blocks):
                    dW0[i,j,k] += deltas[0][t,i,k] * configs[t,j]/n_configs
        db0 += deltas[0][t]/n_configs
        a_in = a_init[0][t].reshape((1,-1))
        dW1 += np.dot(deltas[1][t], a_in)/n_configs
        db1 += deltas[1][t]/n_configs
    
    #Applies the chain rule to sum the gradient components for the individual weights
    dW0 = chain_rule(dW0,latt_dim,n_plaqs,n_blocks,n_loops)
    
    d_weights = [dW0,dW1]
    d_biases = [db0,db1]
    
    if debug == True:
        return deltas, d_weights, d_biases, a_init, z_init
    else:
        return d_weights, d_biases
    
def epoch(configs,actions,weights,biases,latt_dim,n_plaqs,n_loops,n_blocks,
          debug=False,mode='BP'):
    if mode == 'FD':
        d_weights, d_biases = fd_grad(configs,actions,weights,biases)
    else:
        if debug == True:
            deltas,d_weights,d_biases,a_init,z_init=compute_grad(configs,actions,weights,
                                                                 biases,latt_dim,n_plaqs,
                                                                 n_loops,n_blocks,debug)
        else:
            d_weights, d_biases = compute_grad(configs,actions,weights,biases,latt_dim,
                                           n_plaqs,n_loops,n_blocks)
    new_weights = []
    new_biases = []
    for s in range(0,n_layers):
        weight_temp = weights[s] - learn_rate * d_weights[s]
        bias_temp = biases[s] - learn_rate * d_biases[s]
        new_weights.append(weight_temp)
        new_biases.append(bias_temp)
        
    if debug == True:
        return deltas, d_weights, d_biases, a_init, z_init, new_weights, new_biases
    else:
        return new_weights, new_biases
    
def train(weights,biases,n_epochs,n_keep,val_configs,val_actions,train_configs,train_actions,
          write=False,debug=False,mode='BP'):
    weights_in = weights.copy()
    biases_in = biases.copy()
    run_time = time.ctime()
    
    if write == True:
        os.chdir('Data')
        write_dat = str(run_time)+".txt"
        open("std_vals "+write_dat,"w").close()
        open("b_vals "+write_dat,"w").close()
        open("W_vals "+write_dat,"w").close()
    
    std_vals = np.zeros(n_keep)
    w_all = []
    bias_all = []
    deltas = []
    m = 0
    for n in range(0,n_epochs):
        if debug == False:
            weights_out, biases_out = epoch(train_configs,train_actions,weights_in,biases_in,
                                      latt_dim,n_plaqs,n_loops,n_blocks,mode)
        else:
            ep_out = epoch(train_configs,train_actions,weights_in,biases_in,
                                      latt_dim,n_plaqs,n_loops,n_blocks,debug,mode)
            weights_out = ep_out[5]
            biases_out = ep_out[6]
            w_all.append(weights_out)
            bias_all.append(biases_out)
            deltas.append(ep_out[0])
        if n % int(n_epochs/n_keep) == 0:
            z_out, a_out = feed_through_all(val_configs,weights_in,biases_in,
                                            latt_dim,n_plaqs,n_loops,n_blocks)
            temp_cost = cost(val_actions,a_out[-1])
            std_vals[m] = temp_cost
            m += 1
            t_ep = time.time()
            print('Finished epoch '+str(n)+' in '+str(t_ep-t_start)+' seconds.')
            
            if write == True:
                """
                Writes all weight values to a single file, not separating based on
                uniqueness. Includes weights from both layers.
                """
                std_file = open('std_vals '+write_dat,'a')
                std_file.write(str(n)+' '+str(temp_cost)+'\n')
                std_file.close()
                with open("b_vals "+write_dat,'a') as b_file:
                    b_file.write(str(n))
                    for b in biases_out[0].reshape(-1):
                        b_file.write(' '+str(b))
                    for b in biases_out[1].reshape(-1):
                        b_file.write(' '+str(b))
                    b_file.write('\n')
                with open("W_vals "+write_dat,'a') as W_file:
                    W_file.write(str(n))
                    for w in weights_out[0].reshape(-1):
                        W_file.write(' '+str(w))
                    for w in weights_out[1].reshape(-1):
                        W_file.write(' '+str(w))
                    W_file.write('\n')
        weights_in = weights_out
        biases_in = biases_out
        
    if debug == True:
        return std_vals, w_all, bias_all, deltas
    else:
        return std_vals, weights_in, biases_in
plaqs, actions = read_file('4x4')
#test_plaqs = plaqs[0:10,:]#.reshape((22,-1))

latt_dim = [4,4]
n_plaqs = 12
n_blocks = 1
n_loops = [1,4]

n_params = np.sum(n_loops)+n_plaqs

#Excluding input layer - check because code isn't arbitrary yet
n_layers = 2
learn_rate = 1e-6#Much higher of a learning rate makes training completely random

#Imported from block_param_funcs - defines parameters for a block
a, b1, W1 = param_init(latt_dim,n_plaqs,n_blocks,n_loops)#,var_type='Ones',mult=0.1)
b2, W2 = full_layer_params(1,b1.shape[0]*n_blocks)#,param_type='Ones',scale=0.1)

#Probably inefficient way, but chain_rule makes sure that W1 isn't populated
# mostly by zeros.
W1 = chain_rule(mask_mat(latt_dim,n_plaqs,n_blocks,n_loops,W1),latt_dim,n_plaqs,n_blocks,n_loops)
#M = mask_mat(latt_dim,n_plaqs,n_blocks,n_loops)

w = [W1,W2]
b = [b1,b2]

#dw,db,w,b = epoch(test_plaqs,actions[0:10],weights,biases,latt_dim,n_plaqs,
#                 n_loops,n_blocks,debug=True)

#print('dW0 = \n'+str(dW0[:,:,0]))
n_epochs = 5000
n_keep = 200

train_configs = plaqs[0:500,:]
train_actions = actions[0:500]
val_configs = plaqs[500:600,:]
val_actions = actions[500:600]

#z_all, a_all = feed_through_all(val_configs,weights,biases,latt_dim,n_plaqs,n_loops,n_blocks)
#std_test = cost(val_actions,a_all[1])
#print(std_test)

#test=compute_grad(train_configs,train_actions,w,b,latt_dim,n_plaqs,
#                  n_loops,n_blocks,debug=True)
std_out, weight_out, bias_out = train(w,b,n_epochs,n_keep,val_configs,
                                      val_actions,train_configs,train_actions,
                                      write=True,debug=False,mode='FD')
#fd_test = fd_w1(train_configs,train_actions,w,b)
#print('finite difference: \n'+str(w[1] - learn_rate * fd_test))
#print('back propagation: \n'+str(weight_out[-1][1]))

#plt.figure()
#plt.plot(np.arange(n_keep),std_out)

t_end = time.time()
print('Took '+str(t_end-t_start)+' seconds.')