from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop

import numpy as np
import pandas as pd

def create_model(hp):
  
  input_size = 200
  model = Sequential()
  model.add(Input(shape=(input_size,), name='Input'))
  for i in range(hp.Int('num_layers', 2, 5)):
    model.add(Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=1000,
                                            step=32),
                               activation='relu'))
    #model.add(Dropout(hp.Choice('dropout_rate', values=[0, 0.1, 0.2,0.5])))
  

 
  model.add(Dense(1, activation='linear'))
  opt = RMSprop(lr=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
  model.compile(loss='mean_squared_error', optimizer='RMSprop',metrics=['mean_squared_error'])
  return model 

 
 
	
