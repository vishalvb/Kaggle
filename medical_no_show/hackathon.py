
# coding: utf-8

# In[299]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[300]:


data = pd.read_csv('KaggleV2-May-2016.csv')


# In[301]:


data.head()


# In[302]:


data.info()
#No missing values


# In[303]:


data.describe()
#Age has negative values, need to remove them
#PatiendId and AppointmentID are of no use


# In[304]:


data.columns


# ### Some column name are not spelled correctly, fix them

# In[305]:


data.rename(columns = {'Hipertension': 'Hypertension',
                         'Handcap': 'Handicap'}, inplace = True)


# In[306]:


data.columns


# ### Drop the unnecessary column

# In[307]:


data = data.drop(['PatientId','AppointmentID'], axis = 1)


# In[308]:


data.head()


# In[309]:


data.info()


# In[310]:


data_temp = data.copy


# In[311]:


data.head()


# ### Converting data types of ScheduledDay and Appointment day

# In[312]:


data.ScheduledDay = data.ScheduledDay.apply(np.datetime64)


# In[313]:


data.AppointmentDay = data.AppointmentDay.apply(np.datetime64)


# In[314]:


data.info()


# In[315]:


data.head()


# ### Unique values for each columns

# In[316]:


for col in data:
    print (data[col].unique())


# In[317]:


data.Gender.unique()


# In[318]:


data.Handicap.unique()


# In[319]:


data.rename(columns={'No-show' : 'No_show'}, inplace = True)


# In[320]:


data.No_show.unique()


# In[321]:


data.Age.value_counts()


# In[322]:


data = data[(data.Age >=0) & (data.Age <= 90)]


# In[323]:


data.Age.value_counts()


# In[324]:


plt.rcParams['figure.figsize']=(30,10)
sns.countplot(x='Age',data = data)


# In[325]:


data.head()


# In[326]:


data.tail()


# ### Converting Gender, No_show to int values

# In[327]:


data.Gender = data.Gender.eq('M').mul(1)


# In[328]:


data.head()


# In[329]:


data.No_show = data.No_show.eq('No').mul(1)


# In[330]:


data.head()


# In[331]:


type(data.ScheduledDay)


# In[332]:


data.info()


# In[333]:


data['before_days'] = (data.AppointmentDay - np.asarray(data.ScheduledDay.dt.date))


# In[334]:


data.head()


# In[335]:


data['Sche_day'] = data['ScheduledDay'].dt.date


# In[336]:


data.head()


# In[337]:


data = data.drop('before_days',axis = 1)


# In[338]:


data.head()


# In[339]:


data.info()


# In[340]:


data['Sche_day'] = np.asarray(data['Sche_day'])


# In[341]:


data.info()


# In[342]:


data.head()


# In[343]:


data['before_days'] = data.AppointmentDay  - np.asarray(data.Sche_day)


# In[344]:


data.info()


# In[345]:


import datetime


# In[346]:


type(data)


# In[347]:


plt.plot(data['before_days'])


# In[348]:


sns.countplot(data['Gender'])


# In[349]:


sns.countplot(data['No_show'])


# In[350]:


data.info()


# In[351]:


data_female = data[data['Gender'] == 0]


# In[352]:


data_male = data[data['Gender'] ==1]


# In[353]:


data_female['No_show'].value_counts()


# In[354]:


data_male['No_show'].value_counts()


# In[355]:


data['Neighbourhood'].value_counts()


# In[356]:


data.plot(kind='density',subplots=True,layout=(3,3), sharex=False)


# In[357]:


data.Handicap.value_counts()


# In[358]:


data.Alcoholism.value_counts()


# In[359]:


data.Scholarship.value_counts()


# In[360]:


data_temp = data.copy()


# In[361]:


data.columns


# In[362]:


correlations = data.corr()


# In[363]:


names = ['Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood',
       'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap',
       'SMS_received', 'No_show', 'Sche_day', 'before_days']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[364]:


data.columns


# In[365]:


final_data = data[['Gender', 'Age', 'Neighbourhood',
       'Scholarship', 'Hypertension', 'Diabetes',
       'SMS_received', 'No_show']]


# In[366]:


final_data.info()


# In[367]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# In[368]:


final_X = final_data[['Gender', 'Age', 
       'Scholarship', 'Hypertension', 'Diabetes',
       'SMS_received']]


# In[369]:


final_Y = final_data[['No_show']]


# In[370]:


final_X.head()


# In[371]:


final_Y.head()


# In[372]:


X_train, X_test, y_train, y_test = train_test_split(final_X, final_Y, test_size=0.2)


# In[373]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[374]:


rfe = LogisticRegression()


# In[375]:


rfe.fit(X_train,y_train)


# In[376]:


y_pred = rfe.predict(X_test)


# In[377]:



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[378]:


from sklearn import tree


# In[379]:


model = tree.DecisionTreeClassifier()


# In[380]:


model.fit(X_train,y_train)


# In[381]:


model.score(X_train,y_train)


# In[382]:


predicted = model.predict(X_test)


# In[383]:


print(accuracy_score(y_test,predicted))


# In[384]:


from sklearn.naive_bayes import GaussianNB


# In[385]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[386]:


predicted= model.predict(X_test)


# In[387]:


print(accuracy_score(y_test,predicted))


# In[388]:


from sklearn.ensemble import RandomForestClassifier


# In[389]:


model= RandomForestClassifier()


# In[390]:


model.fit(X_train, y_train)
#Predict Output
predicted= model.predict(X_test)


# In[391]:


print(accuracy_score(y_test,predicted))


# In[392]:


from sklearn.ensemble import GradientBoostingClassifier


# In[393]:


model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)


# In[394]:


model.fit(X_train, y_train)
#Predict Output
predicted= model.predict(X_test)


# In[395]:


print(accuracy_score(y_test,predicted))


# In[396]:


plt.plot(data['Gender'])


# In[397]:


data.columns


# In[398]:


sns.countplot(data.SMS_received)


# In[399]:


len(data[(data['SMS_received'] == 0) & (data['No_show'] == 0)])

