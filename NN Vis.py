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

def read_run(run_time):
    files = os.listdir()
    run_files = []
    for file in files:
        file_in = str(file)
        if run_time in file_in:
            run_files.append(file_in)
    del files
    
    for file_in in run_files:
        file = str(file_in)
        with open(file_in) as f:
            if 'std_vals' in file:
                std_dat = f.readlines()
            elif 'b_vals' in file:
                b_dat = f.readlines()
            elif 'W_vals' in file:
                w_dat = f.readlines()
                
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

def plot_run(run_time,params='All'):
    ep, std, b, w = read_run(run_time)
    plt.figure()
    plt.plot(ep,std,'b.-')
    plt.title('Standard Deviation')
    
    if params == 'All':
        plt.figure()
        for t in range(0,b.shape[1]):
            plt.plot(ep,b[:,t])
        plt.title('Biases')
        
        plt.figure()
        for s in range(0,w.shape[1]):
            plt.plot(ep,w[:,s])
        plt.title('Weights')
        
    elif params == 'None':
        return
        
    else:
        n = params
        plt.figure()
        b_range = min(n,b.shape[1])
        for t in range(0,b_range):
            plt.plot(ep,b[:,t])
        plt.title('Biases')
        
        plt.figure()
        for s in range(0,n):
            plt.plot(ep,w[:,s])
        plt.title('Weights')

run = '16:27:02'

plot_run(run,'None')