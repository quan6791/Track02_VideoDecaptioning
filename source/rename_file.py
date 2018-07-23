#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:33:39 2018

@author: digit
"""

import os
path = '../outputs/B1'
files = os.listdir(path)
i = 1

for file_x in files:
    print (i)
    os.rename(os.path.join(path, file_x), os.path.join(path, file_x[:-4]))
    i = i+1
