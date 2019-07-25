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
from scipy import stats

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

data.columns
data.index

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

# these are the codes for the 100 brands that we have statement data for
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
    
#explained_vars_genpop.to_csv(dir_path + '/0_output/2007_explained_vars_genpop.csv')    
#explained_vars_vip.to_csv(dir_path + '/0_output/2007_explained_vars_vip.csv')

# **********************************************************************************
# Stacking the data and PCA calculations
# **********************************************************************************

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
df1['unique_id'] = df1[['respid','id']].apply(lambda x: '@'.join(x), axis=1)

df_fin = pd.DataFrame()
df_fin['unique_id'] = df1['unique_id']


for j in tqdm(name_templ_genpop):
    l = [j + str(i) for i in indices]
    df = data[l].stack(dropna=False)
    df = pd.Series.to_frame(df)
    df.index.levels[0].name = 'Index'
    df = df.merge(df_base,how='left', left_index = True, right_index = True)
    df = df.rename(columns={0: j, 'respid': 'respid' + j, 'survey_qualified_type': 'survey_qualified_type'+j})
    df['id'+j] = df.index
    df['id'+j] = df['id'+j].apply(lambda x: x[1])
    df['id'+j] = df['id'+j].apply(lambda x: x.replace(j,''))
    df['unique_id'] = df[['respid'+j,'id'+j]].apply(lambda x: '@'.join(x), axis=1)
    df_fin = df_fin.merge(df,how='left', on='unique_id')
    

df_fin.drop(['idinterested_in_watching_r', 'respiddifferentiation_r', 'survey_qualified_typedifferentiation_r', 
             'iddifferentiation_r', 'respidculturally_relevant_r', 'survey_qualified_typeculturally_relevant_r', 
             'idculturally_relevant_r', 'respidsocial_responsibility_r', 'survey_qualified_typesocial_responsibility_r', 
             'idsocial_responsibility_r', 'respidability_to_thrive_r', 'survey_qualified_typeability_to_thrive_r', 
             'idability_to_thrive_r', 'respidemotional_connection_r', 'survey_qualified_typeemotional_connection_r', 
             'idemotional_connection_r'], axis=1, inplace=True)

    
df_fin = df_fin.rename(columns={'respidinterested_in_watching_r': 'respid',	
                                'survey_qualified_typeinterested_in_watching_r': 'survey_qualified_type',
                                'interested_in_watching_r': 'interested_in_watching', 'differentiation_r': 'differentiation',	
                                'culturally_relevant_r': 'culturally_relevant', 'social_responsibility_r': 'social_responsibility',	
                                'ability_to_thrive_r': 'ability_to_thrive',	'emotional_connection_r': 'emotional_connection'})
    
cols = ['unique_id', 'respid', 'survey_qualified_type', 'interested_in_watching', 'differentiation',	
        'culturally_relevant', 'social_responsibility',	'ability_to_thrive', 'emotional_connection']
cols_dat = ['interested_in_watching', 'differentiation',	
        'culturally_relevant', 'social_responsibility',	'ability_to_thrive', 'emotional_connection']

df_fin = df_fin[cols]

df_fin_genpop = df_fin.dropna()

X = df_fin_genpop[cols_dat]
X = np.array(X)
ipca = IncrementalPCA(n_components=1)
ipca.fit(X)
score_col = ipca.transform(X)
float(ipca.explained_variance_ratio_)
'''
df_fin['a'] = X[:,0]
df_fin['b'] = X[:,1]
df_fin['c'] = X[:,2]
df_fin['d'] = X[:,3]
df_fin['e'] = X[:,4]
df_fin['f'] = X[:,5]
'''
df_fin_genpop['pca_score_gen_pop'] = score_col



# Calculations for the VIP sample
cols_everyone = ['unique_id', 'respid', 'survey_qualified_type', 'differentiation',	
        'culturally_relevant', 'social_responsibility',	'ability_to_thrive', 'emotional_connection']
cols_dat_everyone = ['differentiation',	'culturally_relevant', 'social_responsibility',	
            'ability_to_thrive', 'emotional_connection']

df_fin_everyone = df_fin[cols_everyone]

df_fin_everyone = df_fin_everyone.dropna()

df_fin_everyone.columns

X = df_fin_everyone[cols_dat_everyone]
X = np.array(X)
ipca = IncrementalPCA(n_components=1)
ipca.fit(X)
score_col = ipca.transform(X)
float(ipca.explained_variance_ratio_)

df_fin_everyone['pca_score_everyone'] = score_col

#df_fin_genpop.to_csv(dir_path + '/0_output/df_fin_genpop.csv')
#df_fin_everyone.to_csv(dir_path + '/0_output/df_fin_everyone.csv')

df_fin_everyone_recoded = df_fin_everyone.copy()

df_fin_everyone_recoded.columns
# this replaces agree with -1 and disgaree with 1
for col in cols_dat_everyone:
    df_fin_everyone_recoded[col].replace({0:-1, 1:1}, inplace=True)
    

# **********************************************************************************
# Constructing a dataframe that contains the final respondent level scores for each brand
# **********************************************************************************
# finding the original respids
resps = df_fin_everyone['respid'].unique()

# finding the brand codes
brand_codes = df_fin_everyone['unique_id'].apply(lambda x: int(x[x.index('@')+1:])).unique()
brand_codes.sort()

colums = ['respid']
#colums.extend(brandpcascores_names)
scores_everyone = pd.DataFrame(index = resps)
scores_everyone['respid'] = scores_everyone.index

df_fin_everyone['brand_code'] = df_fin_everyone['unique_id'].apply(lambda x: x[x.index('@')+1:])

for br in brand_codes:
    scores_df = df_fin_everyone[['respid','pca_score_everyone']][df_fin_everyone['brand_code'] == str(br)]
    #print(scores_df.shape)
    scores_everyone = scores_everyone.merge(scores_df, how = 'left', on='respid')
    #print(scores_everyone.shape)
    scores_everyone = scores_everyone.rename(columns={'pca_score_everyone': 'brandpcascores_' + str(br)})

scores_everyone.head()

#scores_everyone.to_csv(dir_path + '/0_output/everyone_pca_scores.csv', index=False)


# **********************************************************************************
# Timing calculations
# **********************************************************************************
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
    
#explained_vars_genpop.to_csv(dir_path + '/0_output/2007_explained_vars_genpop.csv')    
#explained_vars_vip.to_csv(dir_path + '/0_output/2007_explained_vars_vip.csv')

# ***********************************************
# Stacking the data
# ***********************************************
general_var_names = ['respid', 'survey_qualified_type']
df_base = data[general_var_names]
df_base.index.name = 'Index'

l1 = ['timing_interested_in_watching_r' + str(i) for i in indices]
l2 = ['timing_differentiation_r' + str(i) for i in indices]
l3 = ['timing_culturally_relevant_r' + str(i) for i in indices]
l4 = ['timing_social_responsibility_r' + str(i) for i in indices]
l5 = ['timing_ability_to_thrive_r' + str(i) for i in indices]
l6 = ['timing_emotional_connection_r' + str(i) for i in indices]

df1 = data[l1].stack(dropna=False)
df1 = pd.Series.to_frame(df1)
df1.index.levels[0].name = 'Index' 
df1 = df1.merge(df_base,how='left', left_index = True, right_index = True)
df1['id'] = df1.index
df1['id'] = df1['id'].apply(lambda x: x[1])
df1['id'] = df1['id'].apply(lambda x: x.replace('timing_interested_in_watching_r',''))
df1['unique_id'] = df1[['respid','id']].apply(lambda x: '@'.join(x), axis=1)

df1.columns

df_fin_timing = pd.DataFrame()
df_fin_timing['unique_id'] = df1['unique_id']

name_timing_vars = ['timing_interested_in_watching_r', 'timing_differentiation_r', 'timing_culturally_relevant_r', 'timing_social_responsibility_r', 'timing_ability_to_thrive_r', 'timing_emotional_connection_r']

for j in tqdm(name_timing_vars):
    l = [j + str(i) for i in indices]
    df = data[l].stack(dropna=False)
    df = pd.Series.to_frame(df)
    df.index.levels[0].name = 'Index'
    df = df.merge(df_base,how='left', left_index = True, right_index = True)
    df = df.rename(columns={0: j, 'respid': 'respid' + j, 'survey_qualified_type': 'survey_qualified_type'+j})
    df['id'+j] = df.index
    df['id'+j] = df['id'+j].apply(lambda x: x[1])
    df['id'+j] = df['id'+j].apply(lambda x: x.replace(j,''))
    df['unique_id'] = df[['respid'+j,'id'+j]].apply(lambda x: '@'.join(x), axis=1)
    df_fin_timing = df_fin_timing.merge(df,how='left', on='unique_id')

df_fin_timing.columns    

df_fin_timing.drop(['idtiming_interested_in_watching_r', 'respidtiming_differentiation_r', 'survey_qualified_typetiming_differentiation_r', 
             'idtiming_differentiation_r', 'respidtiming_culturally_relevant_r', 'survey_qualified_typetiming_culturally_relevant_r', 
             'idtiming_culturally_relevant_r', 'respidtiming_social_responsibility_r', 'survey_qualified_typetiming_social_responsibility_r', 
             'idtiming_social_responsibility_r', 'respidtiming_ability_to_thrive_r', 'survey_qualified_typetiming_ability_to_thrive_r', 
             'idtiming_ability_to_thrive_r', 'respidtiming_emotional_connection_r', 'survey_qualified_typetiming_emotional_connection_r', 
             'idtiming_emotional_connection_r'], axis=1, inplace=True)

    
df_fin_timing = df_fin_timing.rename(columns={'respidtiming_interested_in_watching_r': 'respid',	
                                'survey_qualified_typetiming_interested_in_watching_r': 'survey_qualified_type',
                                'timing_interested_in_watching_r': 'timing_interested_in_watching', 'timing_differentiation_r': 'timing_differentiation',	
                                'timing_culturally_relevant_r': 'timing_culturally_relevant', 'timing_social_responsibility_r': 'timing_social_responsibility',	
                                'timing_ability_to_thrive_r': 'timing_ability_to_thrive',	'timing_emotional_connection_r': 'timing_emotional_connection'})
    
cols = ['unique_id', 'respid', 'survey_qualified_type', 'timing_interested_in_watching', 'timing_differentiation',	
        'timing_culturally_relevant', 'timing_social_responsibility',	'timing_ability_to_thrive', 'timing_emotional_connection']

df_fin_timing = df_fin_timing[cols]

df_fin_timing.head()

# Calculations for the VIP sample
cols_everyone_timing = ['unique_id', 'respid', 'survey_qualified_type', 'timing_differentiation',	
        'timing_culturally_relevant', 'timing_social_responsibility',	'timing_ability_to_thrive', 'timing_emotional_connection']

cols_dat_everyone_timing = ['timing_differentiation', 'timing_culturally_relevant', 'timing_social_responsibility',	
                     'timing_ability_to_thrive', 'timing_emotional_connection']

df_fin_timing_everyone = df_fin_timing[cols_everyone_timing]

df_fin_timing_everyone = df_fin_timing_everyone.dropna()

df_fin_timing_everyone.columns

#df_fin_timing_everyone.to_csv(dir_path + '/0_output/df_fin_timing_everyone.csv')

'''
num_bins = 200
n, bins, patches = plt.hist(list(df_fin_timing_everyone['timing_differentiation']), bins=100, range=(0,10000), facecolor='blue', alpha=0.5)
n, bins, patches = plt.hist(list(df_fin_timing_everyone['timing_culturally_relevant']), bins=100, range=(0,10000), facecolor='blue', alpha=0.5)
n, bins, patches = plt.hist(list(df_fin_timing_everyone['timing_social_responsibility']), bins=100, range=(0,10000), facecolor='blue', alpha=0.5)
n, bins, patches = plt.hist(list(df_fin_timing_everyone['timing_ability_to_thrive']), bins=100, range=(0,10000), facecolor='blue', alpha=0.5)
n, bins, patches = plt.hist(list(df_fin_timing_everyone['timing_emotional_connection']), bins=100, range=(0,10000), facecolor='blue', alpha=0.5)

df_fin_timing_everyone['timing_emotional_connection'].mean()
df_fin_timing_everyone['timing_emotional_connection'].median()
df_fin_timing_everyone['timing_emotional_connection'].mode()
df_fin_timing_everyone['timing_emotional_connection'].std()

len(df_fin_timing_everyone['timing_emotional_connection'][df_fin_timing_everyone['timing_emotional_connection'] > 3000])
len(df_fin_timing_everyone['timing_emotional_connection'])
df_fin_timing_everyone['timing_emotional_connection'].max()

mean: 1248
median: 1052
mode: 926
std: 789

df_fin_timing_everyone['timing_emotional_connection'][df_fin_timing_everyone['timing_emotional_connection'] <= 6000].mean()
df_fin_timing_everyone['timing_emotional_connection'][df_fin_timing_everyone['timing_emotional_connection'] <= 6000].median()
df_fin_timing_everyone['timing_emotional_connection'][df_fin_timing_everyone['timing_emotional_connection'] <= 6000].mode()
df_fin_timing_everyone['timing_emotional_connection'][df_fin_timing_everyone['timing_emotional_connection'] <= 6000].std()
'''

df_fin_timing_everyone.columns

time_avgs = pd.DataFrame(df_fin_timing_everyone.mean()/1000)
time_stds = pd.DataFrame(df_fin_timing_everyone.std()/1000)
time_avgs = time_avgs.rename(columns = {0: 'avg'})
time_stds = time_stds.rename(columns = {0: 'std'})
time_avgs.drop(['survey_qualified_type'], axis=0, inplace=True)
time_stds.drop(['survey_qualified_type'], axis=0, inplace=True)

for i in time_avgs.index:
    time_avgs['avg'].loc[i] = df_fin_timing_everyone[i][df_fin_timing_everyone[i] <= 6000].mean()
    time_stds['std'].loc[i] = df_fin_timing_everyone[i][df_fin_timing_everyone[i] <= 6000].std()

std_thresh_very_slow = 1.26
std_thresh_slow = 0.31
std_thresh_fast = -0.31
std_thresh_very_fast = -1.26

time_avgs['std'] = time_stds['std']

time_avgs['very_slow_threshold'] = time_avgs['avg'] + std_thresh_very_slow*time_stds['std']
time_avgs['slow_threshold'] = time_avgs['avg'] + std_thresh_slow*time_stds['std']
time_avgs['fast_threshold'] = time_avgs['avg'] + std_thresh_fast*time_stds['std']
time_avgs['very_fast_threshold'] = time_avgs['avg'] + std_thresh_very_fast*time_stds['std']


def timing_converter(x, t1, t2, t3, t4):
    if x >= t1:
        x = 0
    elif (x >= t2 and x < t1):
        x = 1
    elif (x > t3 and x < t2):
        x = 2
    elif (x <= t3 and x > t4):
        x = 3
    elif x <= t4:
        x = 4
    return x

df_fin_timing_everyone_recoded = df_fin_timing_everyone.copy()
df_fin_timing_everyone_recoded.columns

cols_dat_everyone_timing = ['timing_differentiation', 'timing_culturally_relevant', 'timing_social_responsibility',	
                     'timing_ability_to_thrive', 'timing_emotional_connection']

for c in cols_dat_everyone_timing:
    t1 = time_avgs['very_slow_threshold'].loc[c]
    t2 = time_avgs['slow_threshold'].loc[c]
    t3 = time_avgs['fast_threshold'].loc[c]
    t4 = time_avgs['very_fast_threshold'].loc[c]
    df_fin_timing_everyone_recoded[c] = df_fin_timing_everyone[c].apply(lambda x: timing_converter(x, t1, t2, t3, t4))

#time_avgs.to_csv(dir_path + '/0_output/time_avgs_stds.csv')
#df_fin_timing_everyone_recoded.to_csv(dir_path + '/0_output/df_fin_timing_everyone_recoded.csv')


# **********************************************************************************
# Statement data and timimg multiplied and PCA calculation
# **********************************************************************************

df_fin_everyone_recoded_data_and_time = df_fin_everyone.copy()

for col in cols_dat_everyone:
    df_fin_everyone_recoded_data_and_time[col] = df_fin_everyone_recoded[col]*df_fin_timing_everyone_recoded['timing_' + col]



X = df_fin_everyone_recoded_data_and_time[cols_dat_everyone]
X = np.array(X)
ipca = IncrementalPCA(n_components=1)
ipca.fit(X)
score_col_combined = ipca.transform(X)
float(ipca.explained_variance_ratio_)

df_fin_everyone_recoded_data_and_time['pca_score_combined_everyone'] = score_col_combined

df_fin_everyone_recoded_data_and_time.drop

#df_fin_everyone_recoded_data_and_time.to_csv(dir_path + '/0_output/df_fin_everyone_recoded_data_and_time.csv')

# **********************************************************************************
# Constructing a dataframe that contains the final respondent level scores for each brand
# **********************************************************************************
# finding the original respids
resps = df_fin_everyone_recoded_data_and_time['respid'].unique()

# finding the brand codes
brand_codes = df_fin_everyone_recoded_data_and_time['unique_id'].apply(lambda x: int(x[x.index('@')+1:])).unique()
brand_codes.sort()

colums = ['respid']
#colums.extend(brandpcascores_names)
scores_combined_everyone = pd.DataFrame(index = resps)
scores_combined_everyone['respid'] = scores_combined_everyone.index

df_fin_everyone_recoded_data_and_time['brand_code'] = df_fin_everyone['unique_id'].apply(lambda x: x[x.index('@')+1:])

for br in brand_codes:
    scores_df = df_fin_everyone_recoded_data_and_time[['respid','pca_score_combined_everyone']][df_fin_everyone_recoded_data_and_time['brand_code'] == str(br)]
    #print(scores_df.shape)
    scores_combined_everyone = scores_combined_everyone.merge(scores_df, how = 'left', on='respid')
    #print(scores_combined_everyone.shape)
    scores_combined_everyone = scores_combined_everyone.rename(columns={'pca_score_combined_everyone': 'brandpca_combinedscores_' + str(br)})

scores_combined_everyone.head()

#scores_combined_everyone.to_csv(dir_path + '/0_output/everyone_combined_pca_scores.csv', index=False)


'''
# some visualization
Y = df_fin_timing_everyone['timing_social_responsibility']*df_fin_everyone_recoded['social_responsibility']
Z = df_fin_timing_everyone['timing_culturally_relevant']*df_fin_everyone_recoded['culturally_relevant']

df_fin_everyone_recoded.columns

YZ = pd.DataFrame({'Y':Y, 'Z':Z})
YZ.shape

YZN = YZ[YZ['Y']>-10000][YZ['Z']>-10000][YZ['Y']<10000][YZ['Z']<10000]

DY = pd.DataFrame(Y)

plt.plot(YZN['Y'],YZN['Z'], 'bo')
'''

# **********************************************************************************
# Correlations of love and statements
# **********************************************************************************

col_names = ['brand_' + str(i) for i in indices]
correlations_love_and_statements = pd.DataFrame(columns = col_names, index = cols_dat_everyone)

for i in indices:
    states = [c + '_r' + str(i) for c in cols_dat_everyone]
    states.append('brand_love_r' + str(i))
    subdata = data[states]
    cor = subdata.corr(method='pearson')
    cor = cor['brand_love_r' + str(i)][:-1]
    cor.index = cols_dat_everyone
    correlations_love_and_statements['brand_' + str(i)] = cor
    #print(cor['brand_love_r' + str(i)][:-1])


correlations_love_and_statements.to_csv(dir_path + '/0_output/correlations_love_and_statements.csv')


data_recoded = data.copy()
#data_recoded[['timing_' + cols_dat_everyone[0] + '_r' + str(i) for i in indices]]
#time_avgs.loc['timing_' + cols_dat_everyone[0]]

for c in cols_dat_everyone:
    selected_cols = [c + '_r' + str(i) for i in indices]
    selected_timing_cols = ['timing_' + c + '_r' + str(i) for i in indices]
    t1 = time_avgs['very_slow_threshold'].loc['timing_' + c]
    t2 = time_avgs['slow_threshold'].loc['timing_' + c]
    t3 = time_avgs['fast_threshold'].loc['timing_' + c]
    t4 = time_avgs['very_fast_threshold'].loc['timing_' + c]
    for col in selected_timing_cols:
        data_recoded[col] = data[col].apply(lambda x: timing_converter(x, t1, t2, t3, t4))
    # this replaces agree with -1 and disgaree with 1
    for col in selected_cols:
        data_recoded[col].replace({0:-1, 1:1}, inplace=True)
        
data_recoded.to_csv(dir_path + '/0_output/data_recoded.csv')

data_recoded_datatime = pd.DataFrame()

for c in cols_dat_everyone:
    selected_cols = [c + '_r' + str(i) for i in indices]
    selected_timing_cols = ['timing_' + c + '_r' + str(i) for i in indices]
    for col in selected_cols:
        data_recoded_datatime['datatime_' + col] = data_recoded[col]*data_recoded['timing_' + col]

love_cols = ['brand_love_r' + str(i) for i in indices]
data_recoded_datatime[love_cols] = data[love_cols]

data_recoded_datatime.to_csv(dir_path + '/0_output/data_recoded_datatime.csv')

col_names = ['brand_' + str(i) for i in indices]
correlations_love_and_statements_datatimes = pd.DataFrame(columns = col_names, index = cols_dat_everyone)


for i in indices:
    states = ['datatime_' + c + '_r' + str(i) for c in cols_dat_everyone]
    states.append('brand_love_r' + str(i))
    subdata = data_recoded_datatime[states]
    cor = subdata.corr(method='pearson')
    cor = cor['brand_love_r' + str(i)][:-1]
    cor.index = cols_dat_everyone
    correlations_love_and_statements_datatimes['brand_' + str(i)] = cor
    #print(cor['brand_love_r' + str(i)][:-1])


correlations_love_and_statements_datatimes.to_csv(dir_path + '/0_output/correlations_love_and_statements_datatimes.csv')

# ***************************************
# Correlations of love and statements (stacked)
# ***************************************
general_vars = ['respid']
df_base_love = data[general_vars]
df_base_love.index.name = 'Index'

ll = ['brand_love_r' + str(i) for i in indices]


love_stacked = data[ll].stack(dropna=False)
love_stacked = pd.Series.to_frame(love_stacked)
love_stacked.index.levels[0].name = 'Index' 
love_stacked = love_stacked.merge(df_base_love,how='left', left_index = True, right_index = True)
love_stacked['id'] = love_stacked.index
love_stacked['id'] = love_stacked['id'].apply(lambda x: x[1])
love_stacked['id'] = love_stacked['id'].apply(lambda x: x.replace('brand_love_r',''))
love_stacked['unique_id'] = love_stacked[['respid','id']].apply(lambda x: '@'.join(x), axis=1)

# *****************
# love 1
love_stacked.rename(columns={0: 'brand_love'}, inplace = True)

#love_stacked.to_csv(dir_path + '/0_output/love_stack.csv')

# love 2
love_stacked_with_nil = love_stacked.copy()
love_stacked_with_nil['brand_love'].replace({0:0, 1:1, 2:np.nan, 3:3}, inplace=True)

# love 3
love_stacked_recoded = love_stacked.copy()
love_stacked_recoded['brand_love'].replace({0:0, 1:1, 2:1, 3:1}, inplace=True)

# love 4
love_stacked_recoded_with_nil = love_stacked.copy()
love_stacked_recoded_with_nil['brand_love'].replace({0:0, 1:1, 2:np.nan, 3:1}, inplace=True)
# *****************


statement_stacked = pd.DataFrame()
recoded_statement_stacked = pd.DataFrame()

for col in cols_dat_everyone:
    statement_stacked[col] = df_fin_everyone[col]
    recoded_statement_stacked[col] = df_fin_everyone_recoded_data_and_time[col]

# *****************
# data 1
statement_stacked['unique_id'] = df_fin_everyone['unique_id']

# data 2
recoded_statement_stacked['unique_id'] = df_fin_everyone_recoded_data_and_time['unique_id']
# *****************


# 1st set
statement_love = statement_stacked.merge(love_stacked[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')
statement_love_with_nil = statement_stacked.merge(love_stacked_with_nil[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')
statement_love_recoded = statement_stacked.merge(love_stacked_recoded[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')
statement_love_recoded_with_nil = statement_stacked.merge(love_stacked_recoded_with_nil[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')

statement_love.to_csv(dir_path + '/0_output/statement_love.csv')
statement_love_with_nil.to_csv(dir_path + '/0_output/statement_love_with_nil.csv')

# 2nd set
recoded_statement_love = recoded_statement_stacked.merge(love_stacked[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')
recoded_statement_love_with_nil = recoded_statement_stacked.merge(love_stacked_with_nil[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')
recoded_statement_love_recoded = recoded_statement_stacked.merge(love_stacked_recoded[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')
recoded_statement_love_recoded_with_nil = recoded_statement_stacked.merge(love_stacked_recoded_with_nil[['unique_id', 'brand_love']], on = 'unique_id', how = 'left')



statement_love.corr(method='pearson')['brand_love'][:-1]
statement_love_with_nil.corr(method='pearson')['brand_love'][:-1]
statement_love_recoded.corr(method='pearson')['brand_love'][:-1]
statement_love_recoded_with_nil.corr(method='pearson')['brand_love'][:-1]

recoded_statement_love.corr(method='pearson')['brand_love'][:-1]
recoded_statement_love_with_nil.corr(method='pearson')['brand_love'][:-1]
recoded_statement_love_recoded.corr(method='pearson')['brand_love'][:-1]
recoded_statement_love_recoded_with_nil.corr(method='pearson')['brand_love'][:-1]

# =============================================================================
# Do we see statistically significant differences in the average speed of response 
# by group (Agency, Advertiser, Consumer)? Or by attribute?
# =============================================================================
df_fin_timing_everyone_no_outlier = df_fin_timing_everyone.copy()

def if_replace(x, n):
    news = [i if i <= n else np.nan for i in x]
    return news

df_fin_timing_everyone_no_outlier[cols_dat_everyone_timing] = df_fin_timing_everyone_no_outlier[cols_dat_everyone_timing].apply(lambda x: if_replace(x, 6000))
df_fin_timing_everyone_no_outlier = df_fin_timing_everyone_no_outlier.dropna(axis=0)
df_fin_timing_everyone_no_outlier.shape

timing_detailed_means = df_fin_timing_everyone_no_outlier.groupby(['survey_qualified_type']).mean()

timing_detailed_means.mean(axis=1)
timing_detailed_means.mean(axis=0)

'''
0=Agency
1=Advertisers
2=Gen-pop sample
'''

agency_timing = np.array(df_fin_timing_everyone_no_outlier[cols_dat_everyone_timing][df_fin_timing_everyone_no_outlier['survey_qualified_type'] == 0]).reshape(-1)
advertisers_timing = np.array(df_fin_timing_everyone_no_outlier[cols_dat_everyone_timing][df_fin_timing_everyone_no_outlier['survey_qualified_type'] == 1]).reshape(-1)
genpop_timing = np.array(df_fin_timing_everyone_no_outlier[cols_dat_everyone_timing][df_fin_timing_everyone_no_outlier['survey_qualified_type'] == 2]).reshape(-1)

# Stat tests by respondent type *****************
agency_timing.mean()
advertisers_timing.mean()
genpop_timing.mean()

stats.ttest_ind(advertisers_timing,agency_timing)
stats.ttest_ind(advertisers_timing,genpop_timing)
stats.ttest_ind(agency_timing,genpop_timing)
#************************************************

for col1 in cols_dat_everyone_timing:
    for col2 in cols_dat_everyone_timing:
        print(col1, ' vs. ', col2)
        set1 = df_fin_timing_everyone_no_outlier[col1]
        set2 = df_fin_timing_everyone_no_outlier[col2]
        print(col1, 'mean: ', set1.mean())
        print(col2, 'mean: ', set2.mean())
        print(stats.ttest_ind(set1,set2))
        
# =============================================================================
# It might be interesting to develop another metric, call it "Fast Agree" 
# where 0 = Very Fast Agree or Fast Agree, 1 = all other responses
# =============================================================================
df_fin_everyone_recoded_data_and_time.columns
df_fin_everyone_recoded_data_and_time_fastagree = df_fin_everyone_recoded_data_and_time.copy()
df_fin_everyone_recoded_data_and_time_fastagree[cols_dat_everyone] = df_fin_everyone_recoded_data_and_time[cols_dat_everyone].replace({-4: 0, -3: 0, -2: 1, -1: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
df_fin_everyone_recoded_data_and_time_fastagree.rename(columns = {'differentiation': 'differentiation_rec',
       'culturally_relevant':'culturally_relevant_rec', 'social_responsibility':'social_responsibility_rec', 'ability_to_thrive':'ability_to_thrive_rec',
       'emotional_connection':'emotional_connection_rec'}, inplace = True)

df_fin_everyone_recoded_data_and_time_fastagree = df_fin_everyone_recoded_data_and_time_fastagree.merge(love_stacked_recoded[['brand_love', 'unique_id']], on = 'unique_id', how = 'left')
df_fin_everyone_recoded_data_and_time_fastagree = df_fin_everyone_recoded_data_and_time_fastagree.merge(df_fin_everyone[['differentiation','culturally_relevant', 'social_responsibility', 'ability_to_thrive','emotional_connection', 'unique_id']], on = 'unique_id', how = 'left')

df_fin_everyone_recoded_data_and_time_fastagree.to_csv(dir_path + '/0_output/df_fin_everyone_recoded_data_and_time_fastagree.csv')

