# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:30:12 2019

@author: Tony Yang
@email: t2yang@eng.ucsd.edu
@github: erza0211064@gmail.com
Description:
    Class Lidar_temporal implement temporal_filter, which returns the median of the current and previous d scans.
You can add more scan in one object. Multiple scan data in one object is allow.
Library used:
    numpy: may be use in this problem
    random: only for generate testing data
"""
import random
import numpy as np


class Lidar_temporal:
    def __init__(self,lidar = np.zeros((0,0))):
        '''
        Input:
            lidar: lidar data Type: numpy.ndarray, 1*N, where N is the data length between [200,1000]
        Output:
            self.lidar: lidar data. Type: numpy.ndarray, 1*N
            self.lidarLen: number of lidar data in buffer currently. Type: int
            self.N: length of data. Type: int, between [200,1000]
        '''
        #--check if input data is valid
        if len(lidar.shape) != 2:
            raise Exception("Must be 2D array with (x,N). x is number of scan.")
        if np.any(lidar < 0.03) or np.any(lidar > 50):
            raise Exception("Range of lidar data must be between [0.03,50]")
        if lidar.shape[1] < 200 or lidar.shape[1] > 1000:
            raise Exception("Length of data must be between [200,1000]")
        self.lidar = lidar
        self.lidarLen = self.lidar.shape[0]
        self.N = self.lidar.shape[1]
    def temporal_filter(self,d):
        '''
        Input:
            d: number of previous scan, Type: int
        Output:
            res: lidar data after range filter. Type:numpy.ndarray, 1*N
        '''
        #-- take all lidar data if len <= d
        if self.lidarLen <= d:
            res = np.median(self.lidar, axis = 0)
            return res
        #-- condsider only previous d lidar data
        else:            
            tmp = self.lidar[self.lidarLen - d:self.lidarLen,:]
            res = np.median(tmp, axis = 0)
            return res
    def add_lidar(self, new_lidar):
        '''
        Input:
            new_lidar: new lidar data. Type: numpy.ndarray, 1*N, N must match previous lidar data length
        Output:
            self.lidar: append new lidar data. Type: numpy.ndarray, self.lidarLen*N
            self.lidarLen: number of lidar data in buffer currently. Type: int
        '''
        #-- if data length not fit, raise exception
        if new_lidar.shape[1] != self.N:
            raise Exception("The length of new lidar data does not match previous one. Current length:"+str(self.N))
        self.lidar = np.vstack((self.lidar,new_lidar))
        self.lidarLen = self.lidar.shape[0]
    def delete_lidar(self):
        '''
        Input:
            No
        Output:
            self.lidar: clear all data. Type: numpy.ndarray
            self.lidarLen: 0. Type: int
        '''
        self.lidar = np.zeros((0,0))
        self.lidarLen = 0
    # For debug
    def print_data(self):
        print(len(self.lidar))
        print(self.lidar.shape[0])
        print(self.N)
    def get_lidar(self):
        return self.lidar
    
#--testing
if __name__ == "__main__": 
    #--generate data
    print("generate data...")
    N = 500 # number of lidar data in one scan
    data_num = 10 # number of total lidar data scan
    lidar_list = [] # all lidar test data
    test_range = [] # result after range filter
    test_temporal = [] # result after temporal filter
    for i in range(data_num):
        a = [random.uniform(0.03,50) for i in range(N)]
        lidar = np.array((a))
        lidar_list.append(lidar)
    lidar_list = np.array((lidar_list))
    print("data generate complete...")

    #--test case for temporal, one scan at a time
    print("test temporal...")
    test2 = lidar_list.copy()
    l2 = Lidar_temporal(lidar = test2[0,:].reshape(1,-1))
    test_temporal.append(l2.temporal_filter(5))
    for i in range(1,data_num):
        l2.add_lidar(test2[i,:].reshape(1,-1))
        test_temporal.append(l2.temporal_filter(5))
    test_temporal = np.array((test_temporal))
    #--test case for temporal, all scan at a time
    print("test temporal...")
    test3 = lidar_list.copy()
    l3 = Lidar_temporal(lidar = test3)
    test_temporal2 = l3.temporal_filter(5).reshape(1,-1)

