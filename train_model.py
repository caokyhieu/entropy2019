import keras.backend as K
import tensorflow as tf

import numpy as np
import pickle
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D, MaxPooling2D,AveragePooling2D
from helper import sharpe_ratio, sharpe_ratio_loss
from tuning_model import *


def build_resnet_model(params):


    conv1_ksize = params['filters_1']
    conv1_nfilter = params['filters']

    kernel_size_1 = params['repetitions_1']
    kernel_size_2 = params['repetitions_3']
    kernel_size_3 = params['repetitions_5']
    kernel_size_4 = params['repetitions_7']


    num_filter_1 = params['filters_2']
    num_filter_2 = params['filters_3']
    num_filter_3 = params['filters_4']
    num_filter_4 = params['filters_5']


    reps_1 = params['repetitions']
    reps_2 = params['repetitions_2']
    reps_3 = params['repetitions_4']
    reps_4 = params['repetitions_6']

    conv2_nfilter = params['filters_6']


    regularized_coff_1 = params['l2']
    regularized_coff_2 = params['l2_1']
    regularized_coff_3 = params['l2_2']
    learning_rate = params['l2_3']

    input = Input(shape=(438,64,2))
    conv1 = conv_bn_relu(filters=conv1_nfilter,kernel_size=(1,conv1_ksize),strides=(1,1),\
                         kernel_regularizer=regularizers.l2(regularized_coff_1)) (input)

    pool1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding="same")(conv1)

    out = residual_block(filters=num_filter_1, repetitions=reps_1 ,kernel_size=(1,kernel_size_1),\
                         strides=(1,2),is_first_layer=True) (pool1)

    out = residual_block(filters=num_filter_2, repetitions=reps_2,\
                         kernel_size=(1,kernel_size_2), strides=(1,2)) (out)

    out = residual_block(filters=num_filter_3, repetitions=reps_3,\
                         kernel_size=(1,kernel_size_3),strides=(1,2)) (out)

    out = residual_block(filters=num_filter_4, repetitions=reps_4,\
                         kernel_size=(1,kernel_size_4),strides=(1,2)) (out)

    out = bn_relu(out)

    conv2 = conv_bn_relu(filters=conv2_nfilter,kernel_size=(438,1),strides=(1,1),\
                     kernel_regularizer=regularizers.l2(regularized_coff_2),padding='valid') (out)

    out_shape = K.int_shape(conv2)
    out = AveragePooling2D(pool_size=(out_shape[1], out_shape[2]),
                                 strides=(1, 1))(conv2)

    out = Flatten()(out)

    out = Dense(438, kernel_regularizer =regularizers.l2(regularized_coff_3))(out)
    out = Activation('sigmoid')(out)


    model = Model([input], [out])
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])

    model.summary()
    return model

if __name__ == '__main__':
    ### load hyperparams for models
    with open('model/best_model_params.pkl','rb') as f:

        hyper_params,custom_objects = pickle.load(f)


    X_train,y_train,X_test,y_test = data()
    model = build_resnet_model(hyper_params)
    es = EarlyStopping(monitor='val_sharpe_ratio', mode='max', verbose=1, patience = 30)
    mc = ModelCheckpoint('res_model.h5', monitor='val_sharpe_ratio', mode='max', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=16, epochs= 1000,validation_data=(X_test,y_test),callbacks=[es,mc])
