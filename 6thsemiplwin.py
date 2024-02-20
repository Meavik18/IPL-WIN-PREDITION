#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

#plotting
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#model building
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[6]:


import pandas as pd
import numpy as np

#plotting
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#model building
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[7]:


matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
matches.head()


# In[8]:


deliveries.head()


# In[9]:


matches.shape,deliveries.shape


# In[10]:


matches.columns


# In[11]:


deliveries.columns


# In[12]:


len(matches)


# In[13]:


len(deliveries)


# In[14]:


matches.info()


# In[15]:


deliveries.info()


# In[16]:


matches.isnull().sum()


# In[17]:


deliveries.isnull().sum()


# In[18]:


list1 = matches.columns.to_list()
remove_from_list = ['id', 'date', 'toss_winner', 'toss_decision', 'winner', 
                    'win_by_runs', 'player_of_match', 'venue',
                    'umpire1', 'umpire2', 'umpire3']

for i in range(len(remove_from_list)):
    list1.remove(remove_from_list[i])


# In[19]:


list2 = deliveries.columns.to_list()
remove_from_list2 = ['match_id', 'batsman','inning', 'non_striker', 
                     'bowler', 'player_dismissed', 'fielder']

for i in range(len(remove_from_list2)):
    list2.remove(remove_from_list2[i])
for i in list2:
    print('The unique values in', i, 'are: ', deliveries[i].unique())


# In[20]:


totalrun_df=deliveries.groupby(['match_id','inning']).sum()['total_runs'].reset_index()

totalrun_df


# In[21]:


totalrun_df = totalrun_df[totalrun_df['inning']==1]
totalrun_df['total_runs'] = totalrun_df['total_runs'].apply(lambda x:x+1)#to get target
totalrun_df


# In[22]:


match_df = matches.merge(totalrun_df[['match_id','total_runs']],
                       left_on='id',right_on='match_id')

match_df


# In[23]:


match_df['team1'].unique()


# In[24]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[25]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

# replacing the Deccan Chargers with Sunrises Hyderabad

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[26]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df['team1'].unique()


# In[27]:


match_df[match_df['dl_applied']==1].style.background_gradient(cmap = 'plasma')


# In[28]:


match_df = match_df[match_df['dl_applied']==0]

# considering the match_id, city, winner, and total runs

match_df = match_df[['match_id','city','winner','total_runs']]

match_df


# In[29]:


delivery_df = match_df.merge(deliveries,on='match_id')

delivery_df.head(5)


# In[30]:


delivery_df.columns


# In[31]:


delivery_df.shape


# In[32]:


delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']

delivery_df.head()


# In[33]:


delivery_df['runs_left'] = delivery_df['total_runs_x']-delivery_df['current_score']

delivery_df[['total_runs_x', 'current_score', 'runs_left']].head()


# In[34]:


delivery_df['balls_left'] = 126-(delivery_df['over']*6+delivery_df['ball'])

delivery_df[['over', 'ball', 'balls_left']].head(10)


# In[35]:


delivery_df['player_dismissed']


# In[36]:


# filling nan values with "0"

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")

# now we will convert this player_dismissed col into a boolean col
# if the player is not dismissed then it's 0 else it's 1

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x
                                                                      if x=="0" else "1")

# converting string to int

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')


# In[37]:


delivery_df['player_dismissed'].unique()


# In[38]:


wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values

delivery_df['wickets_left'] = 10-wickets


# In[39]:


delivery_df['cur_run_rate'] = (delivery_df['current_score']*6)/(120-delivery_df['balls_left'])

# required Run-Rate

delivery_df['req_run_rate'] = (delivery_df['runs_left']*6)/(delivery_df['balls_left'])

#Current Run-Rate
delivery_df[['cur_run_rate', 'req_run_rate']].head(10)


# In[40]:


def resultofmatch(row):
    
    return 1 if row['batting_team'] == row['winner'] else 0
    
delivery_df['result'] = delivery_df.apply(resultofmatch,axis=1)


# In[41]:


sns.countplot(delivery_df['result'])


# In[42]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left',
                        'balls_left','wickets_left','total_runs_x','cur_run_rate',
                        'req_run_rate','result']]
                        
                        ##we are taking only important columns 

final_df.head()


# In[ ]:





# In[43]:


final_df.shape


# In[44]:


final_df.dropna(inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


data = final_df.copy()

test = data['result']

train = data.drop(['result'],axis = 1)

train.head()


# In[46]:


final_df.shape


# In[47]:


final_df.isnull().sum()


# In[48]:


final_df = final_df.dropna()

final_df.isnull().sum()


# In[49]:


final_df = final_df[final_df['balls_left'] != 0]


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,test,test_size=0.2,random_state=1)

X_train.shape,X_test.shape


# In[51]:


data = final_df.copy()

test = data['result']

train = data.drop(['result'],axis = 1)

train.head()


# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,test,test_size=0.2,random_state=1)

X_train.shape,X_test.shape


# In[53]:


X_train.columns


# In[54]:


cf = ColumnTransformer(transformers = [
    ('tnf1',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
],remainder='passthrough')



# In[55]:


pipe = Pipeline(steps=[
    ('step1', cf),
    ('step2',LogisticRegression(solver='liblinear'))
])

# fitting the training data

pipe.fit(X_train,y_train)


# In[56]:


pipe = Pipeline(steps=[
    ('step1', cf),
    ('step2',LogisticRegression(solver='liblinear'))
])

# fitting the training data

pipe.fit(X_train,y_train)


# In[57]:


import sklearn.metrics as metrics
y_pred = pipe.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[58]:


pipe.predict_proba(X_test)[10]


# In[59]:


import sklearn.metrics as metrics


# In[60]:


pipe2 = Pipeline(steps=[
    ('step1', cf),
    ('step2',RandomForestClassifier())
])

pipe2.fit(X_train,y_train)
print(metrics.accuracy_score(y_test,pipe2.predict(X_test)))


# In[61]:


pipe2.predict_proba(X_test)[10]


# In[62]:


import pickle
pickle.dump(pipe, open('avikghosh.pkl', 'wb'))


# In[ ]:





# In[ ]:




