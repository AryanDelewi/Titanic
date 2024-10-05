## Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Load in the data
df_train = pd.read_csv('/Users/aryandelewi/Desktop/Git projects/Titanic/Data/titanic/train.csv')

# EDA

## Print head
df_train.head()
print('Survival is our outcome')

## Number of missing values per column
df_train.isna().sum()
(df_train.isna().sum() / df_train.isna().count()) * 100

print("""
        Age missing 177, or 19.8% of the data \n
        Cabin is missing 687 or 77.1% of the data \n
        Embarked is missing 2 or 0.2% of the data \n
      """)

## Descriptive statistics
df_train.describe()

# The name column contains the name but also the title of the individual (i.e. Mr. Miss. etc...)
df_train['Name'].head(20)

#Some tickets have a prefix, lets see what this might be
getprefix = lambda x: x[0] if len(x) > 1 else ''
df_train['Ticket_prefix'] = df_train['Ticket'].str.split(' ').apply(getprefix)

df_train['Ticket_prefix'].value_counts()
#Relatively more males then females
sex_dist = df_train['Sex'].value_counts()
plt.bar(sex_dist.index, sex_dist)
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.title('Count of the gender of the individual')
plt.show()


def hist_outcome_cont(df, var):
    plt.hist(df[df['Survived']==0][var],bins=25,color='red', label='Dead', alpha = 0.5, density=True)
    plt.hist(df[df['Survived']==1][var],bins=25,color='green', label='Alive', alpha = 0.5, density=True)
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.title(f'{var} distribution per survival outcome')
    plt.legend()
    plt.show()
    return None
df_train['SibSp'].value_counts().sort_index()
# Probability of younger passengers to survive is relatively higher
hist_outcome_cont(df_train, 'Age')


# Cant derive a conclusion from the graph
hist_outcome_cont(df_train, 'Fare')

def proba_perclass(df, var = ''):
    proba_outcome_var = (df[df['Survived'] == 1].groupby(var)['Name'].count() / df.groupby(var)['Name'].count()).rename('Probabilty of survival')
    plt.bar(proba_outcome_var.index, proba_outcome_var*100)
    if var =='Pclass':
        plt.xticks([1,2,3],['First class','Second class','Third class'])
    plt.xlabel(var)
    plt.ylabel('Empirical probability of survival (%)')
    plt.title(f'Empirical probability of survival per {var}')
    plt.show()
    return None

## Pclass
#Higher class ticket -> higher probability
proba_perclass(df_train, 'Pclass')

## Title / sex

df_train['Title'] = df_train['Name'].str.split(',').apply(lambda x: x[1]).str.lstrip().str.split(' ').apply(lambda x: x[0])
#Notice how majority is Mr. Miss. Mrs. Master or something rare. 
df_train['Title'].value_counts()
df_train['Adjusted Title'] = np.where(df_train['Title'].isin(['Mr.','Miss.','Mrs.','Master.']),df_train['Title'],'Remaining')
# Married and non married is a potential feature for females? Males would always be 0

# Being a male in general means lower probability of survival?
proba_perclass(df_train, 'Adjusted Title')
#Similar when plotted for Sex
proba_perclass(df_train,'Sex')

## Embarked

#Slight difference in survival probability for embarking
proba_perclass(df_train,'Embarked')
# Most embarked at S. so fill the missing values with S most likely
df_train['Embarked'].value_counts()


# Having one sibling or spouse has a higher probability then the other cases
proba_perclass(df_train,'SibSp')

# Having one sibling or spouse has a higher probability then the other cases
proba_perclass(df_train,'Parch')


# Based on the ticket purchaser, if people are related in any way the probability is higher 
df_train['TicketPurchaser'] = df_train['Name'].str.split('.').apply(lambda x: x[-1]) \
                                              .str.strip().str.split('(').apply(lambda x: x[0]) \
                                              .str.split("\"").apply(lambda x: x[0]) \
                                              .str.strip()

df_TicketPurch = df_train[[
    'Name','Ticket','TicketPurchaser','Age','Sex'
    ]].rename(columns={
    'Name':'Companion_Name',
    'Age':'Companion_Age',
    'Sex':'Companion_Sex'})

df_TicketPurch['Related'] = 1

df_findRelations = df_train.merge(df_TicketPurch, on = ['Ticket','TicketPurchaser'],how = 'left')#,how='left')
df_findRelations['Name'] == df_findRelations['Companion_Name']
cond_diffName = (df_findRelations['Name'] != df_findRelations['Companion_Name'])| df_findRelations['Companion_Name'].isna()
df_findRelations['Related'] = np.where(cond_diffName,1,0)
df_findRelations['Companion_Age'] = np.where(cond_diffName,df_findRelations['Companion_Age'] ,np.nan)
df_findRelations['Companion_Sex'] = np.where(cond_diffName,df_findRelations['Companion_Sex'] ,np.nan)
df_findRelations['Companion_Name'] = np.where(cond_diffName,df_findRelations['Companion_Name'] ,np.nan)

df_findRelations_uniq = df_findRelations.sort_values(['PassengerId','Companion_Name']).groupby(['PassengerId']).first()

proba_perclass(df_findRelations_uniq,'Related')
df_train = df_findRelations_uniq.copy()

#################################################################################################################
##### Impute based on companion if no parentchild relation is appearent, rest with mean based on pclass and title
#################################################################################################################
# if age is missing and parent/child is non existent  fill with companion
df_train['Age_fix1'] = np.where((df_train['Parch'] == 0) & (df_train['Age'].isna()), df_train['Companion_Age'], df_train['Age'])
# Fill the rest of Age as if its the average per sex and Pclass
df_train['Age_fix1'] = df_train.groupby(['Pclass','Adjusted Title'])['Age_fix1'].transform('mean')


#################################################################################################################
##### Impute based on companion if no parentchild relation is appearent, rest with mean based on pclass and title
#################################################################################################################





