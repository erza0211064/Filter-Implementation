# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:30:12 2019

@author: Tony Yang
@email: t2yang@eng.ucsd.edu
@github: erza0211064@gmail.com
Description:
    Class Lidar_range deal with range filter, which crops all lidar data between given minimum and maximum.
There will be only one scan in each object.
Library used:
    numpy: may be use in this problem
    random: only for generate testing data
"""
import random
import numpy as np


class Lidar_range:
    def __init__(self,lidar = np.zeros((0,0))):
        '''
        Input:
            lidar: lidar data Type: numpy.ndarray, 1*N, where N is the data length between [200,1000]
        Output:
            self.lidar: lidar data. Type: numpy.ndarray, 1*N
            self.N: length of data. Type: int, between [200,1000]
        '''
        #--check if input data is valid
        if len(lidar.shape) != 2:
            raise Exception("Must be 2D array with (1,N)")
        if np.any(lidar < 0.03) or np.any(lidar > 50):
            raise Exception("Range of lidar data must be between [0.03,50]")
        if lidar.shape[1] < 200 or lidar.shape[1] > 1000:
            raise Exception("Length of data must be between [200,1000]")
        self.lidar = lidar
        self.N = self.lidar.shape[1]
    def range_filter(self, Min, Max):
        '''
        input:
            Min: minimum lidar range. Type:int, between [0.03,50]
            Max: maximum lidar range. Type:int, between [0.03,50]
        Output:
            res: lidar data after range filter. Type:numpy.ndarray, 1*N
        '''
        res = self.lidar
        res[res >= Max] = Max
        res[res <= Min] = Min
        return res.ravel()
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
    for i in range(data_num):
        a = [random.uniform(0.03,50) for i in range(N)]
        lidar = np.array((a))
        lidar_list.append(lidar)
    lidar_list = np.array((lidar_list))
    print("data generate complete...")

    #--test case for range
    print("test range...")
    test1 = lidar_list.copy()
    for i in range(data_num):
        l1 = Lidar_range(lidar =  test1[i,:].reshape(1,-1))
        test_range.append(l1.range_filter(Min=5,Max=49))
    test_range = np.array((test_range))
