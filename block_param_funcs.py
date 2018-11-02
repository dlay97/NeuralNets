#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 11:40:26 2018

@author: daniellay1
"""
import numpy as np

"""
Deals with funkiness that occurs from the block structure.
"""
# Takes all of the independent parameters in the different block matrices and
# unrolls them into full block matrices. Note that this is NOT the single weight
# matrix.
def unwrap_weights(weights,latt_dim,n_plaqs,n_loops,n_blocks):
    n_t, n_x = latt_dim
    n_x_plaqs = n_x - 1 #Consider defining below, and feeding into the function.
    
    W_in = weights[0]
    X1_in = weights[1]
    X2_in = weights[2]
    Y_in = weights[3]
    U1_in = weights[5]
    U2_in = weights[6]
    V1_in = weights[7]
    V2_in = weights[8]
    
    W_out = np.zeros(shape=(n_plaqs,n_plaqs,n_blocks))
    X1_out = np.zeros(shape=(n_plaqs,n_t,n_blocks))
    X2_out = np.zeros(shape=(n_plaqs,n_t,n_blocks))
    Y_out = np.zeros(shape=(n_loops[1],n_loops[1],n_blocks))
    Z_out = weights[4]
    U1_out = np.zeros(shape=(n_plaqs,n_loops[0],n_blocks))
    U2_out = np.zeros(shape=(n_loops[0],n_plaqs,n_blocks))
    V1_out = np.zeros(shape=(n_loops[1],n_loops[0],n_blocks))
    V2_out = np.zeros(shape=(n_loops[0],n_loops[1],n_blocks))
    
    for k in range(0,n_blocks):
        for i in range(0,n_plaqs):
            if i%n_x_plaqs==0:
                W_out[i,:,k] = np.roll(W_in[0,:,k],i)
            elif i%n_x_plaqs==1:
                W_out[i,:,k] = np.roll(W_in[1,:,k],i-1)
            elif i%n_x_plaqs==2:
                W_out[i,:,k] = np.roll(W_in[2,:,k],i-2)
        for j in range(0,n_t):
            X1_out[:,j,k] = np.roll(X1_in[:,k],n_x_plaqs * j)
            X2_out[:,j,k] = np.roll(X2_in[:,k],n_x_plaqs*j)
        for t in range(0,n_loops[1]):
            Y_out[t,:,k] = np.roll(Y_in[:,k],t)
        for s in range(0,n_plaqs):
            U1_out[s,:,k] = U1_in[s%n_x_plaqs,:,k]
            U2_out[:,s,k] = U2_in[:,s%n_x_plaqs,k]
        V1_out[:,:,k] = V1_in[:,k] * np.ones(shape=n_loops[0])
        V2_out[:,:,k] = V2_in[:,k] * np.ones(shape=n_loops[0])
    
    X2_out = np.transpose(X2_out,axes=(1,0,2))
    
    weights_out = (W_out,X1_out,X2_out,Y_out,Z_out,U1_out,U2_out,V1_out,V2_out)
    
    return weights_out

# Takes in the full weight matrix and returns the block matrices. Note that this
# does NOT give back just the independent parameters.
def block_mat(mat,latt_dim,n_plaqs,n_loops):
    W_out = mat[0:n_plaqs,0:n_plaqs,:]
    X1_out = mat[0:n_plaqs,n_plaqs:n_plaqs+n_loops[1],:]
    X2_out = mat[n_plaqs:n_plaqs+n_loops[1],0:n_plaqs,:]
    Y_out = mat[n_plaqs:n_plaqs+n_loops[1],n_plaqs:n_plaqs+n_loops[1],:]
    Z_out = np.reshape(mat[-1,-1,:],(1,1,-1))
    U1_out = mat[0:n_plaqs,-1,:].reshape((n_plaqs,n_loops[0],-1))
    U2_out = mat[-1,0:n_plaqs,:].reshape((n_loops[0],n_plaqs,-1))
    V1_out = mat[n_plaqs:n_plaqs+n_loops[1],-1,:].reshape((n_loops[1],n_loops[0],-1))
    V2_out = mat[-1,n_plaqs:n_plaqs+n_loops[1],:].reshape((n_loops[0],n_loops[1],-1))
    return (W_out,X1_out,X2_out,Y_out,Z_out,U1_out,U2_out,V1_out,V2_out)

# This takes the block matrices and wrap-sums them into the individual parameters
# used in computing the gradient. This is required because, due to the chain rule,
# dE/dW_{ij} is a sum of all of the shifted terms.
def wrap_sum_weights(weights,latt_dim,n_plaqs,n_loops,n_blocks):
    n_t, n_x = latt_dim
    n_x_plaqs = n_x - 1
    
    W_out = np.zeros((n_x_plaqs,n_plaqs,n_blocks))
    X1_out = np.zeros((n_plaqs,n_blocks))
    X2_out = np.zeros((n_plaqs,n_blocks))
    Y_out = np.zeros((n_loops[1],n_blocks))
    Z_out = weights[4]
    U1_out = np.zeros((n_x_plaqs,n_loops[0],n_blocks))
    U2_out = np.zeros((n_loops[0],n_x_plaqs,n_blocks))
    V1_out = np.reshape(np.asarray(np.sum(weights[7],axis=0)),(n_loops[0],n_blocks))
    V2_out = np.reshape(np.asarray(np.sum(weights[8],axis=1)),(n_loops[0],n_blocks))
    
    for k in range(0,n_blocks):
        for i in range(0,n_plaqs):
            if i%n_x_plaqs==0:
                W_out[0,:,k] += np.roll(weights[0][i,:,k],n_plaqs-i)
            elif i%n_x_plaqs==1:
                W_out[1,:,k] += np.roll(weights[0][i,:,k],n_plaqs+1-i)
            elif i%n_x_plaqs==2:
                W_out[2,:,k] += np.roll(weights[0][i,:,k],n_plaqs+2-i)
        for j in range(0,n_t):
            X1_out[:,k] += np.roll(weights[1][:,j,k],-j*n_x_plaqs)
            X2_out[:,k] += np.roll(weights[2][j,:,k],-j*n_x_plaqs)
        for l in range(0,n_loops[1]):
            Y_out[:,k] += np.roll(weights[3][l,:,k],-l)
        for m in range(0,n_x_plaqs):
            for m1 in range(0,n_plaqs):
                if m1%n_x_plaqs == m%n_x_plaqs:
                    U1_out[m,:,k] += weights[5][m1,:,k]
                    U2_out[:,m,k] += weights[6][:,m1,k]
            
    return (W_out,X1_out,X2_out,Y_out,Z_out,U1_out,U2_out,V1_out,V2_out)

# Combines the block matrices into a single matrix, to play nice with computing
# the gradient (so it fits in a single loop).
def full_mat(weights_in,latt_dim,n_plaqs,n_loops,n_blocks,unwrap=False):
    if unwrap==True:
        weights=unwrap_weights(weights_in,latt_dim,n_plaqs,n_loops,n_blocks)
    else:
        weights=weights_in
    mat_1 = np.concatenate((weights[0],weights[1],weights[5]),axis=1)
    mat_2 = np.concatenate((weights[2],weights[3],weights[7]),axis=1)
    mat_3 = np.concatenate((weights[6],weights[8],weights[4]),axis=1)
    return np.concatenate((mat_1,mat_2,mat_3),axis=0)

# Takes in the wrapped up a vector and outputs the full 17-component vector.
def unwrap_a(a_wrap,latt_dim,n_loops,n_plaqs,n_params):
    a_out = np.zeros(n_params)
    for i in range(0,n_plaqs):
        for j in range(0,latt_dim[1]-1):
            if i%(latt_dim[1]-1)==j:
                a_out[i] = a_wrap[j]
    a_out[n_plaqs:n_plaqs + n_loops[1]] = a_wrap[latt_dim[1]-1] * np.ones(n_loops[1])
    a_out[-1] = a_wrap[-1]
    return a_out

# Takes in the wrapped up b vector and outputs the full 17-component vector.
def unwrap_b(b_wrap,latt_dim,n_loops,n_plaqs,n_params,n_blocks):
    b_out = np.zeros((n_params,n_blocks))
    for i in range(0,n_plaqs):
        for j in range(0,latt_dim[1]-1):
            if i%(latt_dim[1]-1)==j:
                b_out[i,:] = b_wrap[j,:]
    for k in range(0,n_blocks):
        b_out[n_plaqs:n_plaqs + n_loops[1],k] = b_wrap[latt_dim[1]-1,k] * np.ones(n_loops[1])
    b_out[-1,:] = b_wrap[-1,:]
    return b_out

"""
Constructs parameters for the RBM.
"""
# Function for constructing the parameters that will be varied.
def param_init(latt_dim,n_plaqs,n_blocks,n_loops,var_type='Random',r_cent=0,r_std=0.1,mult=1,
               file_name='None'):
    n_t_loops, n_x_loops = n_loops
    n_params = n_plaqs + np.sum(n_loops)
    if var_type == "Random":
        a = np.random.normal(r_cent,r_std,size=(latt_dim[1]+1))
        a = unwrap_a(a,latt_dim,n_loops,n_plaqs,n_params)
        
        b = np.random.normal(r_cent,r_std,size=(latt_dim[1]+1,n_blocks))
        b = unwrap_b(b,latt_dim,n_loops,n_plaqs,n_params,n_blocks)
        
        W = np.random.normal(0,0.1,size=(latt_dim[1]-1,n_plaqs,n_blocks))
        X1 = np.random.normal(0,0.1,size=(n_plaqs,n_blocks))
        X2 = np.random.normal(0,0.1,size=(n_plaqs,n_blocks))
        Y = np.random.normal(0,0.1,size=(n_x_loops,n_blocks))
        Z = np.random.normal(0,0.1,size=(n_t_loops,n_t_loops,n_blocks))
        U1 = np.random.normal(0,0.1,size=(latt_dim[1]-1,n_t_loops,n_blocks))
        U2 = np.random.normal(0,0.1,size=(n_t_loops,latt_dim[1]-1,n_blocks))
        V1 = np.random.normal(0,0.1,size=(n_t_loops,n_blocks))
        V2 = np.random.normal(0,0.1,size=(n_t_loops,n_blocks))
        weights = (W,X1,X2,Y,Z,U1,U2,V1,V2)
    elif var_type == "Ones":
        a = mult*np.ones(shape = latt_dim[1]+1)
        a = unwrap_a(a,latt_dim,n_loops,n_plaqs,n_params)
        
        b = mult*np.ones(shape = (latt_dim[1]+1,n_blocks))
        b = unwrap_b(b,latt_dim,n_loops,n_plaqs,n_params,n_blocks)
        
        W = mult*np.ones(shape=(latt_dim[1]-1,n_plaqs,n_blocks))
        X1 = mult*np.ones(shape=(n_plaqs,n_blocks))
        X2 = mult*np.ones(shape=(n_plaqs,n_blocks))
        Y = mult*np.ones(shape=(n_x_loops,n_blocks))
        Z = mult*np.ones(shape=(n_t_loops,n_t_loops,n_blocks))
        U1 = mult*np.ones(shape=(latt_dim[1]-1,n_t_loops,n_blocks))
        U2 = mult*np.ones(shape=(n_t_loops,latt_dim[1]-1,n_blocks))
        V1 = mult*np.ones(shape=(n_t_loops,n_blocks))
        V2 = mult*np.ones(shape=(n_t_loops,n_blocks))
        weights = (W,X1,X2,Y,Z,U1,U2,V1,V2)
    elif var_type == "File": #Currently reads in one block data and outputs 2 block weights
        os.chdir(os.getcwd()+'/Block Data/')
        with open("a_vals "+file_name,'r') as a_file:
            a_vals = a_file.readlines()
        a = np.asarray(a_vals[-1].split(" ")[1:-1]).astype(np.float)
        with open("b_vals "+file_name,'r') as b_file:
            b_vals = b_file.readlines()
        b = np.asarray(b_vals[-1].split(" ")[1:-1]).astype(np.float)
        b = np.concatenate((b,np.zeros(shape=(1))))
        with open("W_vals[:,0] "+file_name,'r') as weight_file:
            weights = weight_file.readlines()
        weights = np.asarray(weights[-1].split(" ")[1:-1]).astype(np.float)
        W = weights[0:(latt_dim[1]-1)*n_plaqs].reshape((latt_dim[1]-1,n_plaqs,1))
        weights = weights[(latt_dim[1]-1)*n_plaqs:]
        X1 = np.expand_dims(weights[:n_plaqs],axis=-1)
        weights = weights[n_plaqs:]
        U1 = weights[:(latt_dim[1]-1)].reshape((latt_dim[1]-1,1,1))
        weights = weights[(latt_dim[1]-1):]
        X2 = np.expand_dims(weights[:n_plaqs],axis=-1)
        weights = weights[n_plaqs:]
        Y = np.expand_dims(weights[:latt_dim[0]],axis=-1)
        weights = weights[latt_dim[0]:] #Note that this isn't quite as general as it should be!
        V1 = weights[0].reshape((1,1))
        weights = weights[1:]
        U2 = weights[:(latt_dim[1]-1)].reshape((1,latt_dim[1]-1,1))
        weights = weights[(latt_dim[1]-1):]
        V2 = weights[0].reshape((1,1))
        weights = weights[1:]
        Z = weights[0].reshape((1,1,1))
        weights_temp = (W,X1,X2,Y,Z,U1,U2,V1,V2)
        weights = []
        for i in range(0,9):
            temp_var = np.zeros(shape=weights_temp[i].shape)
            weight_i = np.concatenate((weights_temp[i],temp_var),axis=-1)
            weights.append(weight_i)
    elif var_type == "Order":
        a = mult*np.ones(shape = latt_dim[1]+1)
        a = unwrap_a(a,latt_dim,n_loops,n_plaqs,n_params)
        
        b = mult*np.ones(shape = (latt_dim[1]+1,n_blocks))
        b = unwrap_b(b,latt_dim,n_loops,n_plaqs,n_params,n_blocks)
        
        init_val = 0
        fin_val = (latt_dim[1]-1)*n_plaqs
        W = 0.1*np.arange(init_val,fin_val).reshape((latt_dim[1]-1,n_plaqs,n_blocks))
        init_val += fin_val
        fin_val = n_plaqs
        X1 = 0.1*np.arange(init_val,init_val + fin_val).reshape((n_plaqs,n_blocks))
        init_val += fin_val
        fin_val = latt_dim[1]-1
        U1 = 0.1*np.arange(init_val,init_val+fin_val).reshape((latt_dim[1]-1,n_t_loops,n_blocks))
        init_val += fin_val
        fin_val = n_plaqs
        X2 = 0.1*np.arange(init_val,init_val+fin_val).reshape((n_plaqs,n_blocks))
        init_val += fin_val
        fin_val = latt_dim[1]
        Y = 0.1*np.arange(init_val,init_val+fin_val).reshape((n_x_loops,n_blocks))
        init_val += fin_val
        fin_val = n_t_loops
        V1 = 0.1*np.arange(init_val,init_val+fin_val).reshape((n_t_loops,n_blocks))
        init_val += fin_val
        fin_val = latt_dim[1]-1
        U2 = 0.1*np.arange(init_val,init_val+fin_val).reshape((n_t_loops,latt_dim[1]-1,n_blocks))
        init_val += fin_val
        fin_val = n_t_loops
        V2 = 0.1*np.arange(init_val,init_val+fin_val).reshape((n_t_loops,n_blocks))
        init_val += fin_val
        fin_val = n_t_loops
        Z = 0.1*np.arange(init_val,init_val+fin_val).reshape((n_t_loops,n_t_loops,n_blocks))
        
        weights = (W,X1,X2,Y,Z,U1,U2,V1,V2)
    else:
        print("Pick an optional parameter type.")
        return 'Nan'
    return a, b, weights

def mask_mat(latt_dim,n_plaqs,n_blocks,n_loops,weights='None'):
    if type(weights)==str:
        a, b, weights = param_init(latt_dim,n_plaqs,n_blocks,n_loops,var_type='Ones')
    
    W_in = weights[0]
    W_out = np.zeros((n_plaqs,n_plaqs,n_blocks))
    W_out[0:W_in.shape[0],0:W_in.shape[1],0:W_in.shape[2]] = W_in
    
    X1_in = weights[1]
    X1_out = np.zeros((n_plaqs,n_loops[1],n_blocks))
    X1_out[0:X1_in.shape[0],0,0:X1_in.shape[1]] = X1_in
    
    X2_in = weights[2]
    X2_out = np.zeros((n_loops[1],n_plaqs,n_blocks))
    X2_out[0,0:X2_in.shape[0],0:X2_in.shape[1]] = X2_in
    
    Y_in = weights[3]
    Y_out = np.zeros((n_loops[1],n_loops[1],n_blocks))
    Y_out[0,0:Y_in.shape[0],0:Y_in.shape[1]] = Y_in
    
    Z_out = weights[4]
    
    U1_in = weights[5]
    U1_out = np.zeros((n_plaqs,n_loops[0],n_blocks))
    U1_out[0:U1_in.shape[0],0:U1_in.shape[1],0:U1_in.shape[2]] = U1_in
    
    U2_in = weights[6]
    U2_out = np.zeros((n_loops[0],n_plaqs,n_blocks))
    U2_out[0:U2_in.shape[0],0:U2_in.shape[1],0:U2_in.shape[2]] = U2_in
    
    V1_in = weights[7]
    V1_out = np.zeros((n_loops[1],n_loops[0],n_blocks))
    V1_out[0:V1_in.shape[0],0,0:V1_in.shape[1]] = V1_in
    
    V2_in = weights[8]
    V2_out = np.zeros((n_loops[0],n_loops[1],n_blocks))
    V2_out[0,0:V2_in.shape[0],0:V2_in.shape[1]] = V2_in
    
    weights = (W_out,X1_out,X2_out,Y_out,Z_out,U1_out,U2_out,V1_out,V2_out)
    
    weight_mat = full_mat(weights,latt_dim,n_plaqs,n_loops,n_blocks)
    
    return weight_mat

#Applies the chain rule for the derivative in a block matrix
def chain_rule(weight_mat,latt_dim,n_plaqs,n_blocks,n_loops):
    weight_in = block_mat(weight_mat,latt_dim,n_plaqs,n_loops)
    weight_mid = wrap_sum_weights(weight_in,latt_dim,n_plaqs,n_loops,n_blocks)
    weight_out = full_mat(weight_mid,latt_dim,n_plaqs,n_loops,n_blocks,unwrap=True)
    return weight_out

#Applies the chain rule for the fully connected layer
def chainRuleF(weights,latt_dim,n_plaqs,n_blocks,n_loops):
    n_distinct = int(n_plaqs/latt_dim[0])#number of distinct plaquettes
    weight_out = np.zeros(weights.shape)
    for t in range(0,n_blocks):
        weight_temp = np.zeros(np.sum(n_loops))
        for s in range(0,latt_dim[0]):
            weight_temp[0:n_distinct] += weights[s*n_distinct:(s+1)*n_distinct,t]
        weight_temp[n_distinct] = np.sum(weights[n_plaqs:n_plaqs+latt_dim[0],t])
        weight_temp[-1] = weights[-1,t]
        
        for s in range(0,latt_dim[0]):
            weight_out[s*n_distinct:(s+1)*n_distinct,t] = weight_temp[0:n_distinct]
        weight_out[n_plaqs:n_plaqs+latt_dim[0],t] = weight_temp[n_distinct]
        weight_out[-1,t] = weight_temp[-1]
    return weight_out
        


"""
Fully connected layer stuff.

Since the output is a number, the bias should just be a number.
"""
def fnn_param_init(latt_dim,n_blocks,param_type='Rand',loc=0,scale=1):
    in_size = latt_dim[0] + 1#latt_dim[0]-1 for plaqs, +1 for t_loops, +1 for x_loop
            #also - loops may be flipped
    weight_out = np.zeros((np.prod(latt_dim)+1,n_blocks))
    for t in range(0,n_blocks):
        if param_type == 'Zeros':
            weights = np.zeros((in_size))
            biases = np.zeros(1)
        elif param_type == 'Rand':
            weights = np.random.normal(loc,scale,size=(in_size))
            biases = np.random.normal(loc,scale,size=1)
        elif param_type == 'Ones':
            weights = scale*np.ones((in_size))
            biases = scale*np.ones(1)
        else:
            print('Pick a layer type.')
            return
        for i in range(0,latt_dim[1]):
            weight_out[i*(latt_dim[0]-1):(i+1)*(latt_dim[0]-1),t] = weights[0:latt_dim[0]-1]
        weight_out[(latt_dim[0]-1)*latt_dim[1]:,t] = weights[latt_dim[0]-1]
        weight_out[-1,t] = weights[-1]
    return biases, weight_out
    