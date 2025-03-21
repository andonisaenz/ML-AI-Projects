# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:17:47 2025

@author: Andoni Sáenz
"""

#%% IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% UPLOAD THE DATASET

df = pd.read_csv("AbandonoEmpleados.csv", sep = ";")
print(df.info())
print(df.describe())
print(df.shape)