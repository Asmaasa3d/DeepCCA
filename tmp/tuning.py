from DNN import create_model
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import numpy as np
import pandas as pd

from get_data import main
data =main()

train_set_x, train_set_y = data[0]
valid_set_x, valid_set_y = data[1]
test_set_x, test_set_y = data[2]

tuner = kt.Hyperband(create_model,
                     objective='val_loss',
                     max_epochs=20,
                     factor=3,
                     directory='my_dir',
                     project_name='contact_tracing')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tuner.search_space_summary()
tuner.search(train_set_x, train_set_y, epochs=50, validation_data=(valid_set_x, valid_set_y), callbacks=[stop_early])

best_model=tuner.get_best_hyperparameters()[0]
print(best_model.values)

hypermodel = tuner.hypermodel.build(best_model)


history = hypermodel.fit(train_set_x, train_set_y, epochs=50, validation_data=(valid_set_x, valid_set_y ))

ypred = hypermodel.predict(test_set_x)
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
hypermodel.summary()
