#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:42:02 2024

@author: kalyani
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sales = pd.read_csv('./sales_dataset/Advertising.csv')

X = sales.iloc[:, 0:4]
Y = sales.iloc[:, 4:]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

plt.plot(X, Y, 'b.', label="Advertising")
plt.xlabel("Media")
plt.ylabel("Sales")
plt.show()

deg = 6;
ply_ftres = PolynomialFeatures(degree=deg, include_bias=False)
x_poly = ply_ftres.fit_transform(x_train)

ply_reg = LinearRegression()
ply_reg.fit(x_poly, y_train)

x_tst_ply = ply_ftres.transform(x_test)
y_pred = ply_reg.predict(x_tst_ply)

plt.plot(np.sort(x_test, axis=0), y_pred[np.argsort(x_test, axis=0)[:, 0]], color='red')
plt.show()