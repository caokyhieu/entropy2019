import pandas as pd
import numpy as np
import datetime

import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from tuning_model import *
import pickle
from helper import sharpe_ratio, sharpe_ratio_loss
from simple_rest_client.api import API
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def prepair_input_data(data_path= 'data/data.csv',end_date='2019-07-31',timesteps=65):
    df = pd.read_csv(data_path)
    df['date'] = df.date.apply(pd.Timestamp)
    df['dow'] = df.date.apply(lambda x: x.dayofweek)
    ## just select working days
    df = df[(df.dow<=4)&(df.dow>=0)]
    df = df.drop(['dow'],axis=1)


    df = df.pivot(index='date', columns='ticker')
    window_stop_row = df[df.index<=end_date].iloc[-1]
    iloc_stop = df.index.get_loc(window_stop_row.name)
    X = df.iloc[iloc_stop+1-timesteps:iloc_stop+1].values
    input_data = X.reshape(X.shape[0],2,-1)[np.newaxis]
    ##fillna
    X[np.isnan(X)] = 0.0
    input_data = np.moveaxis(input_data,-1,1)
    return input_data, df.columns.get_level_values(1)[:438].values

def load_model_by_name(name):
    PATH_MODEL ='model'
    with open(PATH_MODEL +'/' + name +'_params.pkl','rb') as f:
        hyper_params,custom_objects = pickle.load(f)

    model = load_model(PATH_MODEL+'/'+name+'.h5',custom_objects=custom_objects)
    return model,hyper_params,custom_objects



def call_api_sharpe_ratio(tickers):
    api = API(api_root_url='http://128.199.65.170:5000', timeout=60)
    api.add_resource(resource_name='entropy2019')
    api.entropy2019.actions
    response = api.entropy2019.list(body=None, params={'tickers': ','.join([ticker.lower() for ticker in tickers])}, headers={})
    return response.body['sharpe_ratio']


def predict_tickers(model, data_path= 'data/data.csv', end_date= '2019-07-31'):
    df = pd.read_csv(data_path)
    input_data, ticker_array = prepair_input_data(data_path,end_date=end_date,timesteps=64)
    y_predict = model.predict(input_data)
    available_tickers = df[df['date'] == end_date]['ticker'].values
    ticker_dict = {}
    full_tickers = {t:v for t,v in zip(ticker_array, y_predict.reshape((-1,)))}
    for t,v in full_tickers.items() :
        if (t in available_tickers):
            ticker_dict[t] = v
    score_topK = sorted(ticker_dict.items(), key = lambda kv:-kv[1])[:50]
    tickers = [v for v,i in score_topK if i>0.5]
    return tickers,ticker_dict,full_tickers

def main():
    model,hyper_params,custom_objects = load_model_by_name('best_model')
    data_path = 'data/data.csv'
    end_date = '2019-07-31'
    tickers,_,_ = predict_tickers(model, data_path, end_date)
    score = call_api_sharpe_ratio(tickers)
    print("################## selected tickers################")
    print(tickers)
    print("Sharpe Ratio = %0.5f"%(float(score)))
    print()
    print()

    print('Export file result.csv')
    result = pd.DataFrame({'ticker':tickers})
    result.to_csv(path_or_buf='result_E-L9362_CAOKYHIEU.csv',index=False)
    print('File had been export')

main()
