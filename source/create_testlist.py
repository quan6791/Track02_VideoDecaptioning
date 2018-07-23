#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:41:00 2018

@author: digit
"""

import numpy as np
import os
import glob
from pandas.io.parsers import read_csv
import h5py
import sys
import matplotlib.pyplot as plt
import scipy.misc as scm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


arr=[]
path_file = '../data_set/test/X/*.mp4'
for filename in sorted(glob.glob(path_file)):
    arr.append(filename)
    #print (filename)
arr = np.asarray(arr)

np.savetxt("final_test.txt",arr,fmt='%s');
