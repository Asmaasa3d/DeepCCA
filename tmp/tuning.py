from DNN import create_model
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

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
tuner.search(train_set_x, train_set_y, epochs=50, validation_data=[valid_set_x, valid_set_y ], callbacks=[stop_early])

best_model=tuner.get_best_hyperparameters()[0].values
print(best_model)

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.summary()
