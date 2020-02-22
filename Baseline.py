#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
import os


# In[2]:


os.chdir('D:\Github\google NCAA\Results')


# In[3]:


tourney_result = pd.read_csv("D:\Datasets\Google_NCAA\MNCAATourneyCompactResults.csv")


# In[4]:


tourney_seed = pd.read_csv("D:\Datasets\Google_NCAA\MNCAATourneySeeds.csv")


# In[5]:


tourney_result


# In[6]:


tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)


# In[7]:


tourney_seed


# In[8]:


tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], 
                          right_on=['Season', 'TeamID'], 
                          how='left')


# In[9]:


tourney_result


# In[10]:


tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)


# In[11]:


tourney_result = tourney_result.drop('TeamID', axis=1)


# In[12]:


tourney_result


# In[13]:


tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], 
                          right_on=['Season', 'TeamID'], 
                          how='left')
tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


# In[14]:


tourney_result


# In[15]:


def get_seed(x):
    return int(x[1:3])

tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))
tourney_result


# In[16]:


season_result = pd.read_csv("D:\Datasets\Google_NCAA\MRegularSeasonCompactResults.csv")


# In[17]:


season_result


# In[18]:


season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
season_win_result


# In[19]:


season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
season_lose_result


# In[20]:


season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)
season_result


# In[21]:


season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
season_score


# In[22]:


tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], 
                          right_on=['Season', 'TeamID'], 
                          how='left')


# In[23]:


tourney_result


# In[24]:


tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)


# In[25]:


tourney_result = tourney_result.drop('TeamID', axis=1)


# In[26]:


tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], 
                          right_on=['Season', 'TeamID'], 
                          how='left')


# In[27]:


tourney_result


# In[28]:


tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result


# In[29]:


tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)
tourney_win_result


# In[30]:


tourney_lose_result = tourney_win_result.copy()
tourney_lose_result['Seed1'] = tourney_win_result['Seed2']
tourney_lose_result['Seed2'] = tourney_win_result['Seed1']
tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']
tourney_lose_result


# In[31]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']


# In[32]:


tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
tourney_result


# In[33]:


test_df = pd.read_csv("D:\Datasets\Google_NCAA\MSampleSubmissionStage1_2020.csv")


# In[34]:


test_df


# In[35]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
test_df


# In[36]:


test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df


# In[37]:


test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
test_df


# In[38]:


X = tourney_result.drop('result', axis=1)
y = tourney_result.result


# In[39]:


result_table = pd.DataFrame(columns=['num_leaves','min_data_in_leaf','objective','max_depth','learning_rate','boosting_type',
                                    'bagging_seed','metric','verbosity','random_state','min_child_weight',
                                    'feature_fraction','bagging_fraction','reg_alpha','reg_lambda','fold','log_loss',
                                    'overall_log_loss'])


# In[58]:


from sklearn.model_selection import KFold
import lightgbm as lgb

params = {'num_leaves': 100,
          'min_data_in_leaf': 20,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 7,
          "metric": 'logloss',
          "verbosity": -1,
          'random_state': 42,
         }


# In[59]:


for i in range(10):
    result_table = result_table.append(params, ignore_index=True)


# In[60]:


import gc
NFOLDS = 10
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=17)


# In[61]:


columns = X.columns


# In[62]:


splits = folds.split(X, y)


# In[63]:


y_preds = np.zeros(test_df.shape[0])
y_preds


# In[64]:


y_oof = np.zeros(X.shape[0])
y_oof


# In[65]:


y_valid_coll = np.zeros(X.shape[0])


# In[66]:


feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
feature_importances


# In[67]:


for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    pd.options.mode.chained_assignment = None
    result_table['fold'].loc[fold_n] = fold_n+1
    
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    y_valid_coll[valid_index] = y_valid
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval = 4)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    
    print(round((log_loss(y_valid, y_pred_valid)),2))
    result_table['log_loss'].loc[fold_n] = log_loss(y_valid, y_pred_valid)
               
    y_oof[valid_index] = y_pred_valid
        
    y_preds += clf.predict(test_df) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()

print("log loss:",round(log_loss(y_valid_coll, y_oof),2))
result_table = result_table.append({'overall_log_loss':log_loss(y_valid_coll, y_oof)}, ignore_index=True)


# In[68]:


result_table.to_csv('result_table.csv',index=False,  mode='a', header=False)


# In[69]:


submission_df = pd.read_csv("D:\Datasets\Google_NCAA\MSampleSubmissionStage1_2020.csv")
submission_df['Pred'] = y_preds
submission_df


# In[70]:


submission_df['Pred'].hist()


# In[71]:


submission_df.to_csv('submission.csv', index=False)


# In[72]:


import seaborn as sns
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(6, 6))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[ ]:




