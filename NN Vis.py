#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:55:30 2018

@author: daniellay1
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.getcwd()+'/Plaqs/Data')

def read_run(run_time,dat_type = 'Plaqs'):
    files = os.listdir()
    run_files = []
    for file in files:
        file_in = str(file)
        if run_time in file_in:
            run_files.append(file_in)
    del files
    
    for file_in in run_files:
        file = str(file_in)
        if dat_type == 'Plaqs':
            with open(file_in) as f:
                if 'std_vals' in file:
                    std_dat = f.readlines()
                elif 'b_vals' in file:
                    b_dat = f.readlines()
                elif 'W_vals' in file:
                    w_dat = f.readlines()
        elif dat_type == 'Configs':
            with open(file_in) as f:
                if 'Bias' in file:
                    b_dat = f.readlines()
                elif 'Weight' in file:
                    w_dat = f.readlines()
                elif 'Cost' in file:
                    std_dat = f.readlines()#Not really standard deviation right now
                
    n_eps = len(std_dat)
    epochs = np.zeros(n_eps)
    std_out = np.zeros(n_eps)
    b_out = []
    w_out = []
    
    for t in range(0,n_eps):
        ep_temp, std_temp = std_dat[t].split(' ')#[1].replace('\n','')
        std_temp.replace('\n','')
        epochs[t] = ep_temp
        std_out[t] = std_temp
        
        b_temp = b_dat[t].split(' ')[1:]
        b_temp = [x.replace('\n','') for x in b_temp]
        b_out.append(b_temp)
        
        w_temp = w_dat[t].split(' ')[1:]
        w_temp = [x.replace('\n','') for x in w_temp]
        w_out.append(w_temp)
    
    b_out = np.asarray(b_out).astype(float)
    w_out = np.asarray(w_out).astype(float)
    
    return epochs, std_out, b_out, w_out

def read_legend():
    with open('Legend.txt','r') as leg:
        legend = leg.readlines()
    times = []
    latt_dim = []
    n_blocks = []
    n_dims = []
    adam = []
    for s in range(1,len(legend)):
        temp_key = legend[s].split(', ')
        times.append(temp_key[0])
        latt_dim.append(temp_key[1])
        n_blocks.append(temp_key[2])
        n_dims.append(temp_key[3])
        adam.append(temp_key[4])
        
    times = [time.replace('.txt','') for time in times]
    latt_dim = np.asarray(latt_dim).astype(int)
    n_blocks = np.asarray(n_blocks).astype(int)
    n_dims = np.asarray(n_dims).astype(int)
    adam = [a.replace('\n','') for a in adam]
    
    return times,latt_dim,n_blocks,n_dims,adam

def plot_run(run_time,b_range='Auto',w_range='Auto',params='None',dat_type='Plaqs'):
    ep, std, b, w = read_run(run_time,dat_type)
   
    plt.figure()
    plt.plot(ep[:],std[:],'b.-')
    plt.title('Standard Deviation '+run_time)
    if b_range == 'Auto':
        b_range = b.shape[1]
    if w_range == 'Auto':
        w_range = w.shape[1]
    if params == 'All':
        plt.figure()
        for t in range(0,b_range):
            plt.plot(ep,b[:,t],'.-')
        plt.title('Biases')
        
        plt.figure()
        for s in range(0,w_range):
            plt.plot(ep,w[:,s],'.-')
        plt.title('Weights')
        
    elif params == 'None':
        return ep, std
        
    elif params == 'Weights':        
        plt.figure()
        for s in range(0,w_range):
            plt.plot(ep,w[:,s],'.-')
        plt.title('Weights')
        
    elif params == 'Biases':
        plt.figure()
        for t in range(0,b_range):
            plt.plot(ep,b[:,t],'.-')
        plt.title('Biases')
    return ep, std
        
def plot_min(day_files):
    std_all = np.zeros(0)
    for file in day_files:
        ep, std, b, w = read_run(str(file),'Plaqs')
        std_all = np.append(std_all,std[-1])
    return std_all

def heat_map(time,latt_dim,n_blocks,n_dims,file_type='Plaqs',plot_vals='All',
             var_names=None,t_ep=-1):
    ep, std, b, w = read_run(time,file_type)
    n_params = latt_dim**2 + 1
    w0_size = n_params**2
    
    w0 = w[t_ep,0:n_blocks*w0_size].reshape((n_params,n_params,n_blocks))
    w1 = w[t_ep,n_blocks*w0_size:].reshape((n_params,n_blocks))
    b0 = b[t_ep,0:n_blocks*n_params].reshape((n_params,n_blocks))
    if plot_vals == 'All':
        var_array = [w0,w1,b0]
    elif plot_vals == 'w0':
        var_array = [w0]
    var_names = ['w0','w1','b0']
    for i in range(0,len(var_array)):
        if var_names[i] == 'w1' or var_names[i] == 'b0':
            var_temp = np.concatenate([var_array[i][:,s].reshape((-1,1))\
                                         for s in range(0,n_blocks)],axis=1)
            plt.figure()
            plt.imshow(var_temp)#.reshape((-1,n_blocks)))
            plt.colorbar()
            plt.title(time+' '+var_names[i])
        else:
            for s in range(0,n_blocks):
                plt.figure()
                plt.pcolor(var_array[i][:,:,s],vmin=np.min(w0),vmax=np.max(w0))
                plt.colorbar()
                plt.title(time+' w0('+str(s)+') '+str(t_ep))
                #plt.savefig(time+' '+var_names[i]+' '+str(s))
    return w1

times,latt_dim,n_blocks,n_dims,adam = read_legend()
j = -1
ep, std = plot_run(times[j])
n_eps = len(ep)

w1 = heat_map(times[j],latt_dim[j],n_blocks[j],n_dims[j],plot_vals='w0')