import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import pickle




def prepair_data(path,window_x,window_y):
    df = pd.read_csv(path)
    df['date'] = df.date.apply(pd.Timestamp)

    df['dow'] = df.date.apply(lambda x: x.dayofweek)
    ## just select working days
    df = df[(df.dow<=4)&(df.dow>=0)]
    df = df.drop(['dow'],axis=1)

    df = df.pivot_table(index='date', columns='ticker')

    daily_return = df.close.pct_change()

    tickers = df.close.columns

    X = df.values.reshape(df.shape[0],2,-1)
    y = daily_return.values

    ## fill X
    ##fill nan by 0.0
    X[np.isnan(X)] = 0.0

    ## fill y with -1e2 
    y[np.isnan(y)] = -1e2

    X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
    y = rolling_array(y[window_x:],stepsize=1,window=window_y)
    X = np.moveaxis(X,-1,1)
    y = np.swapaxes(y,1,2)

    return X,y,tickers



def rolling_array(a, stepsize=1, window=60):
    n = a.shape[0]
    return np.stack((a[i:i + window:stepsize] for i in range(0,n - window + 1)),axis=0)




if __name__ == '__main__':
    X,y,tickers = prepair_data('data/data.csv',window_x=64,window_y=19)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    ### saved data
    print('Save data at data folder')
    with open('data/data_used.pkl','wb') as f:
        pickle.dump((X_train,X_val,y_train,y_val),f,protocol=pickle.HIGHEST_PROTOCOL)
