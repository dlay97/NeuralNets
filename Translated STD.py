#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:13:33 2018

@author: daniellay1
"""

import os
import numpy as np
import time

t0 = time.time()

file_name = "4x4"

os.chdir(os.getcwd()+"/Plaqs/")

"""
Computes the plaquettes.
"""

# Wraps a lattice configuration that's indexed according to the QED convention.
def latt_wrap_qed(config,latt_size):
    temp_out = np.zeros(latt_size)
    t_configs = config[0::2]
    x_configs = config[1::2]
    temp_out[:,:,0] = np.reshape(t_configs,latt_size[0:2],order='F')
    temp_out[:,:,1] = np.reshape(x_configs,latt_size[0:2],order='F')
    return temp_out

def plaq(config,latt_size,on_latt='True'):
    if on_latt == "False":
        config = latt_wrap_qed(config,latt_size)
    plaq_out = np.zeros((latt_size[0],latt_size[1]))
    n_t = latt_size[0]
    n_x = latt_size[1]
    for t in range(0,latt_size[0]):
        for x in range(0,latt_size[1]):
            plaq_out[t,x] = config[t,x,1] + config[t,(x+1)%n_x,0] \
                            - config[(t+1)%n_t,x,1] - config[t,x,0]
    plaq_out = np.reshape(plaq_out,(-1,))
    plaq_out = plaq_out[0:-1]
    t_loop = np.sum(config[:,0,0])
    x_loop = np.sum(config[0,:,1])
    plaq_out = np.append(plaq_out,[[t_loop],[x_loop]])
    return plaq_out

"""
Translates a configuration.
"""
def translate(config,latt_dim,n_t,n_x):
    config_in = latt_wrap_qed(config,latt_dim)
    config_out = np.zeros(shape=latt_dim)
    for i in range(0,latt_dim[0]):
        for j in range(0,latt_dim[1]):
            config_out[(i+n_t)%latt_dim[0],(j+n_x)%latt_dim[1],:] = config_in[i,j,:]
    return config_out

"""
Reads in the configs.
"""
def read_configs():
    # Reads in configs from file.
    with open(file_name+" Configs.txt") as file:
        configs = file.readlines()
    
    temp_configs = np.asarray(configs)
    configs = []
    for s in range(0,len(temp_configs)):
        config = temp_configs[s].split(" ")
        temp_config = np.zeros(len(config))
        for i in range(0,len(config)-1):
            temp_config[i] = float(config[i])
        configs.append(temp_config)
    configs = np.asarray(configs)
    configs = configs[0:2000,0:-1]

    return configs
"""
Constructs parameters of RBM.
"""
configs = read_configs()

n_configs, config_len = configs.shape
latt_dim = np.asarray([4,4,2]) #n_time, n_space, n_dimensions (e.g. 1+1, 2+1)
if np.prod(latt_dim) != config_len:
    print('Check the lattice size!!!')
    
#Prints all translations of a particular configuration.
norm_plaq = plaq(configs[0,:],latt_dim,'False')
print("Unshifted plaquette: \n"+str(norm_plaq)+'\n')
for n_t in range(0,latt_dim[0]):
    for n_x in range(0,latt_dim[1]):
        print("n_t: "+str(n_t))
        print("n_x: "+str(n_x))
        trans_config = translate(configs[0,:],latt_dim,n_t,n_x)
        shift_plaq = plaq(trans_config,latt_dim,"True")
        print("Shifted plaquette: \n"+str(shift_plaq)+"\n")


t1 = time.time()
print('This took '+str(t1-t0)+' seconds.')