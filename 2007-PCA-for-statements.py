# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:51:54 2019

@author: payam.bagheri
"""

import pandas as pd
import numpy as np
from os import path
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
plot(X[:,0],X[:,1], 'bo')

ipca = IncrementalPCA(n_components=2, batch_size=3)
ipca.fit(X)
ipca.transform(X) 
ipca.explained_variance_ratio_
'''

dir_path = path.dirname(path.dirname(path.abspath(__file__)))
print(dir_path)
data = pd.read_csv(dir_path + '/0_input_data/2007-statement-timing-raw-data.csv')

samp_cols = ['interested_in_watching_r4', 'differentiation_r4', 'culturally_relevant_r4', 'social_responsibility_r4', 'ability_to_thrive_r4', 'emotional_connection_r4']
samp_data = data[samp_cols]
samp_data = samp_data.fillna(2)
X = np.array(samp_data)

plt.plot(X[:,0],X[:,1], 'bo')

ipca = IncrementalPCA(n_components=1)
ipca.fit(X)
ipca.transform(X) 
ipca.explained_variance_ratio_

name_templ_genpop = ['interested_in_watching_r', 'differentiation_r', 'culturally_relevant_r', 'social_responsibility_r', 'ability_to_thrive_r', 'emotional_connection_r']
indices = [0, 2, 3, 6, 7, 9, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 37, 40, 42, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 61, 63, 64, 65, 68, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99, 102, 104, 105, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 129, 130, 131, 132, 133, 134, 136, 137, 143, 144, 145, 150, 152]
explained_vars_genpop = pd.DataFrame(columns = ['vars'], index=indices)
len(indices)

for i in indices:
    selected_cols = [x+str(i) for x in name_templ_genpop]
    X = data[selected_cols]
    X = X.dropna()
    X = np.array(X)
    ipca = IncrementalPCA(n_components=1)
    ipca.fit(X)
    ipca.transform(X)
    exp_var = float(ipca.explained_variance_ratio_)
    explained_vars_genpop.loc[i] = exp_var

name_templ_vip = ['differentiation_r', 'culturally_relevant_r', 'social_responsibility_r', 'ability_to_thrive_r', 'emotional_connection_r']
indices = [0, 2, 3, 6, 7, 9, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 37, 40, 42, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 61, 63, 64, 65, 68, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99, 102, 104, 105, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 129, 130, 131, 132, 133, 134, 136, 137, 143, 144, 145, 150, 152]
explained_vars_vip = pd.DataFrame(columns = ['vars'], index=indices)
len(indices)

for i in indices:
    selected_cols = [x+str(i) for x in name_templ_vip]
    X = data[selected_cols]
    X = X.dropna()
    X = np.array(X)
    ipca = IncrementalPCA(n_components=1)
    ipca.fit(X)
    ipca.transform(X)
    exp_var = float(ipca.explained_variance_ratio_)
    explained_vars_vip.loc[i] = exp_var
    
num_bins = 20
n, bins, patches = plt.hist(list(explained_vars_genpop['vars']), num_bins, facecolor='blue', alpha=0.5)
n, bins, patches = plt.hist(list(explained_vars_vip['vars']), num_bins, facecolor='blue', alpha=0.5)
    
explained_vars_genpop.to_csv(dir_path + '/0_output/2007_explained_vars_genpop.csv')    
explained_vars_vip.to_csv(dir_path + '/0_output/2007_explained_vars_vip.csv')

general_var_names = ['respid', 'survey_qualified_type']
df_base = data[general_var_names]
df_base.index.name = 'Index'

l1 = ['interested_in_watching_r' + str(i) for i in indices]
l2 = ['differentiation_r' + str(i) for i in indices]
l3 = ['culturally_relevant_r' + str(i) for i in indices]
l4 = ['social_responsibility_r' + str(i) for i in indices]
l5 = ['ability_to_thrive_r' + str(i) for i in indices]
l6 = ['emotional_connection_r' + str(i) for i in indices]

df1 = data[l1].stack(dropna=False)
df1 = pd.Series.to_frame(df1)
df1.index.levels[0].name = 'Index' 
df1 = df1.merge(df_base,how='left', left_index = True, right_index = True)
df1['id'] = df1.index
df1['id'] = df1['id'].apply(lambda x: x[1])
df1['id'] = df1['id'].apply(lambda x: x.replace('interested_in_watching_r',''))
df1['unique_id'] = df1[['respid','id']].apply(lambda x: ''.join(x), axis=1)

df_fin = pd.DataFrame()
df_fin['unique_id'] = df1['unique_id']


for j in tqdm(name_templ_genpop):
    l = [j + str(i) for i in indices]
    df = data[l].stack(dropna=False)
    df = pd.Series.to_frame(df)
    df.index.levels[0].name = 'Index' 
    df = df.merge(df_base,how='left', left_index = True, right_index = True)
    df['id'] = df.index
    df['id'] = df['id'].apply(lambda x: x[1])
    df['id'] = df['id'].apply(lambda x: x.replace(j,''))
    df['unique_id'] = df[['respid','id']].apply(lambda x: ''.join(x), axis=1)
    df_fin = df_fin.merge(df,how='left', on='unique_id')


df_fin.to_csv(dir_path + '/0_output/df_fin.csv')

'''
pcadat = pd.read_csv(dir_path + '/0_output/df_fin.csv')

X = pcadat[[str(i) for i in range(1,7)]]
X = X.dropna()
X = np.array(X)
ipca = IncrementalPCA(n_components=1)
ipca.fit(X)
len(ipca.transform(X))
float(ipca.explained_variance_ratio_)
'''