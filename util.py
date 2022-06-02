import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder


def group_email_domain(df):
    """
    Function that group  email domain bty the service provider  grouped  email domain that
    has very few coiunts into  category 'other'
    :param df:  pandas dataframe
    :return:  df:  pandas dataframe
    """
    for email in ['P_emaildomain', 'R_emaildomain']:
        df.loc[df[email].
                   isin(['gmail.com', 'gmail']), email] = 'Google Mail'
        df.loc[df[email].
                   isin(['yahoo.com', 'ymail.com', 'yahoo.com.mx',
                         'yahoo.co.jp', 'yahoo.fr', 'yahoo.co.uk',
                         'yahoo.es', 'yahoo.de']), email] = 'Yahoo Mail'
        df.loc[df[email].
                   isin(['hotmail.com', 'outlook.com', 'msn.com',
                         'live.com', 'live.com.mx', 'outlook.es',
                         'hotmail.fr', 'hotmail.co.uk', 'live.fr',
                         'hotmail.es', 'hotmail.de']), email] = 'Microsoft mail'
        df.loc[df[email].
                   isin(['icloud.com', 'me.com', 'mac.com']), email] = 'Apple mail'

        df.loc[df[email].
                   isin(df[email].
                        value_counts()[df[email].
                        value_counts() <= 1000].index), email] = 'Others'
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def resumetable(df):
    print(f'Dataset Shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = \
        round(stats.entropy(df[name].value_counts(normalize=True), base=2), 2)
    
    return summary


# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
def count_fraud_plot(df, col):
    """
       Return  a count  distribution plot against categorical column  and the fraud percentage of each category
       :param df:  pandas dataframe
       :param col_name:  str . categorical column
       :return:
       """

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
    sns.countplot(x=col, data=df,
                  order=list(tmp[col].values), ax=ax[0])

    # ax[0].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[0].set_title(
        f"Frequency of {col} values", fontsize=16)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    for p in ax[0].patches:
        height = p.get_height()
        ax[0].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height / df.shape[0] * 100:.2f}%',
                   ha='center', fontsize=8)

    sns.barplot(x=col, y='Fraud', data=tmp,
                order=list(tmp[col].values), ax=ax[1])

    ax[1].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[1].set_title(
        f"Fraud Percentage of {col} values", fontsize=16)
    ax[1].set_ylabel("Percent", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    plt.subplots_adjust(hspace=.4, top=0.9)

    for p in ax[1].patches:
        height = p.get_height()
        ax[1].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height:.2f}%',
                   ha='center', fontsize=8)
    plt.subplots_adjust(hspace=.4, top=1.1)
    plt.show()

def cat_num_features(df):
    
    '''
        Utility Function to get the names of Categorical Features and 
        Numerical Features of the given Dataset.
    '''
    
    catf = []
    numf = []
    
    # Given Categorical Features 
    catf = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', \
            'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', \
            'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', \
            'DeviceType', 'DeviceInfo']
    catf+=['id_'+str(i) for i in range(12,39)]


    # Updating the Categorical Feature Names List based on the columns present in the dataframe
    catf = [feature for feature in catf if feature in df.columns.values]
    numf = [feature for feature in df.columns if feature not in catf and not feature == 'isFraud']
    
    return (catf, numf)  



def label_encode(X_train, X_test, catf):
  
  '''
    Utility Function to Encode Categorical Features.
  '''

  for f in catf:
    
    X_train[f] = X_train[f].astype(str)
    X_test[f] = X_test[f].astype(str)
    
    le = LabelEncoder()
    le.fit(X_train[f])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    X_train[f] = le.transform(X_train[f])
    
    # Manually Encoding the CV and Test Dataset so as to avoid error for any category which is not present in train set
    
    # All the categories which are not present in train datset are encoded as -1
    
    X_test[f] = [-1 if mapping.get(v, -1)==-1 else mapping[v] for v in X_test[f].values ]

  return (X_train, X_test)


def normalize(X_train, X_test):
    '''
        Utility Function to scale the values of the Train, CV and Test Datasets between 0 and 1.
    '''
    
    for f in X_train.columns:

        min_val = X_train[f].min()
        max_val = X_train[f].max()
        
        X_train[f] = (X_train[f]-min_val)/(max_val-min_val)
        X_test[f] = (X_test[f]-min_val)/(max_val-min_val)
        
    return (X_train, X_test)

