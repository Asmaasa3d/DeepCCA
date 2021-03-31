from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd

def create_model(hp):
  
  input_size = data[0][0].shape[1]
  model = Sequential()
  model.add(Input(shape=(input_size,), name='Input'))
  for i in range(hp.Int('num_layers', 2, 5)):
    model.add(Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=1000,
                                            step=32),
                               activation='relu'))
    #model.add(Dropout(hp.Choice('dropout_rate', values=[0, 0.1, 0.2,0.5])))
  

  '''
  model.add(Dense(ls, activation='relu'))
  if dropout:
    model.add(Dropout(dropout))
  model.add(Dense(ls, activation='relu'))
  if dropout:
    model.add(Dropout(dropout))
  model.add(Dense(ls,activation='relu'))
  if dropout:
    model.add(Dropout(dropout))
  model.add(Dense(ls,activation='relu'))

  if dropout:
    model.add(Dropout(dropout))'''
  model.add(Dense(1, activation='linear'))
  opt = RMSprop(lr=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
  model.compile(loss='mean_squared_error', optimizer='RMSprop',metrics=['mean_squared_error'])
  return model 
'''
 model.summary()
 history = model.fit(train_set_x, train_set_y, epochs=epochs, batch_size=batch_size,validation_data=[valid_set_x, valid_set_y ])
  
  
  ypred = model.predict(test_set_x)
  # show the inputs and predicted outputs
  
  #print("Mean_error=%s" % mean_absolute_error(test_set_y, ypred))
  #Min_error=1000
  #Max_error=0
  acc=0
  TH=1.0
  error=[]
  for i in range(len(ypred)):
    error.append(abs(test_set_y[i]-ypred[i]))
    #Min_error=min(Min_error,abs(test_set_y[i]-ypred[i]))
    #Max_error=max(Max_error,abs(test_set_y[i]-ypred[i]))
    if abs(test_set_y[i]-ypred[i])<1.0:
      acc+=1
    #print("actual_Dis=%s, Predicted=%s" % (test_set_y[i],ypred[i]))
  error = np.asarray(error)
  results=[]
  Percentiles=[0,25,50,75,100]
  for  i in Percentiles:
    p= np.percentile(error, i)
    results.append(p)
    #print("Percentile_%s=%s" %  (i,p) )
  print(results)
  print("Accuracy =", (acc/len(ypred)) * 100.0) 
'''
  
	
