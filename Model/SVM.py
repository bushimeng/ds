#!/usr/bin/env python
# coding: utf-8

#Import Required Modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import plotly.graph_objects as go #
import plotly.express as px
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
pallete = ['Accent_r', 'Blues', 'BrBG', 'BrBG_r', 'BuPu', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'OrRd', 'Oranges', 'Paired', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdGy_r', 'RdPu', 'Reds', 'autumn', 'cool', 'coolwarm', 'flag', 'flare', 'gist_rainbow', 'hot', 'magma', 'mako', 'plasma', 'prism', 'rainbow', 'rocket', 'seismic', 'spring', 'summer', 'terrain', 'turbo', 'twilight']
import scipy.stats as stats
%matplotlib inline
from scipy.stats import chi2_contingency
import scipy.stats as stats
import statsmodels.api as sm
import scipy.stats.distributions as dist
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from prettytable import PrettyTable
scaler = MinMaxScaler()


data = '/home/ubuntu/DistributedDreamTeam/data/diabetes.csv'
df = pd.read_csv('diabetes.csv')

df['frame'] = df['frame'].replace({'small': 0, 'medium': 1, 'large': 2})

#X = df.drop(['diabetic','location'], axis=1)
X = df.drop(['diabetic','location','height','bp.1d','time.ppn','gender'], axis=1)
#X
print(X.shape)
y = df['diabetic']
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=333, test_size=0.2, stratify = y )
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)



svc = svm.SVC(random_state=333)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc) * 100
precision_svc = precision_score(y_test, y_pred_svc) * 100
recall_svc = recall_score(y_test, y_pred_svc) * 100
f1_svc = f1_score(y_test, y_pred_svc) * 100
print('\nSupport Vector Machine Performance:')
print('Accuracy:', accuracy_svc)
print('Precision:', precision_svc)
print('Recall:', recall_svc)
print('F1 Score:', f1_svc)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_svc))