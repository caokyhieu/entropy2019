from hyperas.distributions import uniform
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D, MaxPooling2D,AveragePooling2D

import numpy as np
import pickle
from keras import regularizers
from keras.optimizers import Adam,SGD
from keras.models import Model
from helper import *
import pickle

def save_model(model,hyper_params,name,functions_name,functions):
    PATH_MODEL = 'model'
    if not os.path.exists(PATH_MODEL):
        os.makedirs(PATH_MODEL)

    model.save(PATH_MODEL +'/' +  name +'.h5')

    custom_objects = {name:func for name,func in zip(functions_name,functions)}
    with open(PATH_MODEL +'/' + name +'_params.pkl','wb') as f:
        pickle.dump([hyper_params,custom_objects],f,protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved model at %s'%(PATH_MODEL +'/' +  name +'.h5'))
    print('Saved hyperparams at %s'%(PATH_MODEL +'/' + name +'_params.pkl'))

    pass





def data():
    with open('data/data_used.pkl','rb') as f:
        X_train,X_test,y_train,y_test = pickle.load(f)

    return X_train,y_train,X_test,y_test




def bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)


def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return bn_relu(conv)

    return f


def bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    """

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

def short_cut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """

    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=regularizers.l2(0.001))(input)

    return Add()([shortcut, residual])

def residual_block(filters, repetitions,kernel_size=(3,3),strides=(2,2), is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = strides
            input = basic_block(filters=filters,kernel_size=kernel_size, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters,kernel_size=(3,3), init_strides=(1, 1), is_first_block_of_first_layer=False):

    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="glorot_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=kernel_size,
                                  strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=kernel_size)(conv1)
        return short_cut(input, residual)

    return f


def create_resnet_model(X_train, y_train, X_test, y_test):



    input = Input(shape=(438,64,2))
    conv1 = conv_bn_relu(filters={{choice([4, 8, 12])}},kernel_size=(1,{{choice([3,5,7])}}),\
                         strides=(1,1),kernel_regularizer=regularizers.l2({{uniform(1e-5,1e-1)}})) (input)


    pool1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding="same")(conv1)


    out = residual_block(filters={{choice([6,8,10])}}, repetitions={{choice([1,2])}} ,\
                         kernel_size=(1,{{choice([3,5,7])}}),strides=(1,2),is_first_layer=True) (pool1)

    out = residual_block(filters={{choice([6,8, 10])}}, repetitions={{choice([1,2])}},\
                         kernel_size=(1,{{choice([3,5,7])}}),strides=(1,2)) (out)

    out = residual_block(filters={{choice([6, 8,10])}}, repetitions={{choice([1,2])}},\
                         kernel_size=(1,{{choice([3,5,7])}}),strides=(1,2)) (out)

    out = residual_block(filters={{choice([4,5,6])}}, repetitions={{choice([1,2])}},\
                         kernel_size=(1,{{choice([3,5,7])}}),strides=(1,2)) (out)



    out = bn_relu(out)

    conv2 = conv_bn_relu(filters={{choice([6,7,8])}},kernel_size=(438,1),strides=(1,1),\
                     kernel_regularizer=regularizers.l2({{uniform(1e-5,1e-1)}}),padding='valid') (out)

    out_shape = K.int_shape(conv2)

    out = AveragePooling2D(pool_size=(out_shape[1], out_shape[2]),
                                 strides=(1, 1))(conv2)

    out = Flatten()(out)

    out = Dense(438, kernel_regularizer =regularizers.l2({{uniform(1e-5,1e-1)}}))(out)
    out = Activation('sigmoid')(out)


    model = Model([input], [out])
    optimizer = Adam(lr = {{uniform(1e-5,1e-1)}})
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])



    model.fit(X_train, y_train, batch_size={{choice([16,32,64])}}, epochs= 10)
    score = model.evaluate(X_test, y_test, verbose=2)
    sharpe = score[1]
    return {'loss': -sharpe, 'status': STATUS_OK, 'model': model}




if __name__ == '__main__':
    functions = [sharpe_ratio_loss,sharpe_ratio,bn_relu,conv_bn_relu,bn_relu_conv,short_cut,residual_block,basic_block]
    trials = Trials()
    best_run,best_model = optim.minimize(model=create_resnet_model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=10,
                            eval_space=True,
                             functions=functions,
                              trials=trials)
    X_train,y_train,X_test,y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    func_name = ['sharpe_ratio_loss','sharpe_ratio','bn_relu','conv_bn_relu','bn_relu_conv','short_cut','residual_block','basic_block']

    print('Do cross validate model')




    print('Save best model')
    save_model(best_model,best_run,'new_model',func_name,functions)
