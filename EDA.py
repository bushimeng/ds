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



data = '/home/ubuntu/DistributedDreamTeam/data/diabetes.csv'
df = pd.read_csv('diabetes.csv')

#Boxplots
columns_to_exclude = ['location', 'gender', 'frame','diabetic']
filtered_columns = [col for col in df.columns if col not in columns_to_exclude]


plt.figure(figsize=(12, 12))
sns.set(style="whitegrid")
for col in filtered_columns:
    plt.subplot(3, 4, filtered_columns.index(col) + 1)
    sns.boxplot(data=df, y=col, orient="v", width=0.5)
    plt.title(f'Boxplot of {col}')
    plt.savefig("Boxplots.pdf", format="pdf")
    plt.tight_layout()

plt.show()

columns_to_exclude = ['location', 'gender', 'frame', 'diabetic']
filtered_columns = [col for col in df.columns if col not in columns_to_exclude]

plt.figure(figsize=(12, 12))
sns.set(style="whitegrid")

for i, col in enumerate(filtered_columns):
    plt.subplot(3, 4, i + 1)


    sns.boxplot(x="diabetic", y=col, data=df, width=0.5)


    plt.title(f'Boxplot of {col}')
plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.savefig("Boxplots_pair.pdf", format="pdf")

plt.tight_layout()

plt.show()

#Histogram according to Diabetic status
attributes_to_exclude = ['location', 'gender', 'frame']

df_filtered = df.drop(attributes_to_exclude, axis=1)

plt.figure(figsize=(12, 10))
for i, col in enumerate(df_filtered.columns[:-1]):
    plt.subplot(3, 4, i + 1)
    sns.histplot(df_filtered[df_filtered['diabetic'] == 1][col], kde=True, label='Diabetes', color='red', stat = 'density')
    sns.histplot(df_filtered[df_filtered['diabetic'] == 0][col], kde=True, label='No Diabetes', color='green',stat = 'density')
    plt.title(f"Distribution of {col}")
    plt.legend()
plt.tight_layout()
plt.savefig("Histograms.pdf", format="pdf")
plt.show()


#Pairwise Plots
pp=sns.pairplot(df,hue='diabetic')
pp
#pp.figure.savefig("Pairwise_plots.pdf", format="pdf")

kde=sns.pairplot(df, kind="kde")
kde
kde.figure.savefig("kde_plots.pdf", format="pdf")

corrMatrix = df.loc[:, df.columns != 'diabetic'].corr()
sns.clustermap(corrMatrix, annot = True, fmt = ".3f")
plt.title("Correlation Between Features")
plt.savefig("Correlations.pdf", format="pdf", bbox_inches = 'tight')
plt.show()


df.groupby('diabetic').mean().T

df.groupby('diabetic').median().T

df.groupby('diabetic').var().T

df.groupby('diabetic').std().T

for i in df.columns:
    if(i != 'location' and i != 'gender' and i != 'frame'):
      plt.figure()
      stats.probplot(df[i], plot = plt)

      plt.title(i)
      plt.show()

    else:
      print("Type Error")
      
attributes_to_exclude = ['location', 'gender', 'frame']

df_filtered = df.drop(attributes_to_exclude, axis=1)

plt.figure(figsize=(12, 10))
for i, col in enumerate(df_filtered.columns[:-1]):
    plt.subplot(3, 4, i + 1)
    stats.probplot((df_filtered[col]-np.mean(df_filtered[col]))/np.std(df_filtered[col]), plot=plt)
    plt.title(f"QQ-Plot of Standardised {col}")
    plt.ylabel("Sample Quantile")

plt.tight_layout()
plt.savefig("QQPlots.pdf", format="pdf")
plt.show()

from scipy.stats import norm, skew

skewed_feats = df.skew(0).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = ["Skewed"])
skewness.T

df


fig, axes = plt.subplots(3, 1, figsize=(16, 25))

sns.countplot(x='location', hue='diabetic', data=df, ax=axes[0])
axes[0].set_title('Diabetic Status Based on Location', fontsize=20)
axes[0].set_xlabel('Location', fontsize=20)
axes[0].set_ylabel('Count', fontsize=20)
axes[0].legend(title='Diabetic', loc='upper right', fontsize=20)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(x='gender', hue='diabetic', data=df, ax=axes[1])
axes[1].set_title('Diabetic Status Based on Gender', fontsize=20)
axes[1].set_xlabel('Gender', fontsize=20)
axes[1].set_ylabel('Count', fontsize=20)
axes[1].legend(title='Diabetic', loc='upper right', fontsize=20)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

sns.countplot(x='frame', hue='diabetic', data=df, ax=axes[2])
axes[2].set_title('Diabetic Status Based on Frame', fontsize=20)
axes[2].set_xlabel('Frame', fontsize=20)
axes[2].set_ylabel('Count', fontsize=20)
axes[2].legend(title='Diabetic', loc='upper right', fontsize=20)
axes[2].grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Bar_charts.pdf", format="pdf")
plt.show()


specific_attribute = 'stab.glu'
condition_column = 'diabetic'

# Filter rows where diabetic is 0
filtered_df = df[df[condition_column] == 0]

# Find the index of the row with the maximum value in the specified column
max_index = filtered_df[specific_attribute].idxmax()

# Extract the row corresponding to the maximum value
row_with_max_value = df.loc[max_index]

# Print the row
print("Row with the largest value in '{}' where '{} == 0':".format(specific_attribute, condition_column))
print(row_with_max_value)


location_proportions = df.groupby('location')['diabetic'].mean()

frame_proportions = df.groupby('frame')['diabetic'].mean()

gender_proportions = df.groupby('gender')['diabetic'].mean()

# Print results
print("Proportions of Diabetics based on Location:")
print(location_proportions)

print("\nProportions of Diabetics based on Frame:")
print(frame_proportions)

print("\nProportions of Diabetics based on Gender:")
print(gender_proportions)

df.groupby('gender')['diabetic'].sum()

female_count = df[df['gender'] == 'female'].shape[0]
female_count

male_count = df[df['gender'] == 'male'].shape[0]
male_count




