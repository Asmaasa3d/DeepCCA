from keras.optimizers import RMSprop
from keras.regularizers import l2
from objectives import cca_loss

from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Concatenate
from keras.models import Model


def create_model(layer_sizes1, layer_sizes2, input_size1, input_size2,
                    learning_rate, reg_par, outdim_size, use_all_singular_values, dropout=None):

    view1_input = Input(shape=(input_size1,), name='View1_Input')
    view2_input = Input(shape=(input_size2,), name='View2_Input')

    view1_model_layer = build_mlp_net(layer_sizes1, reg_par, view1_input, dropout=dropout)
    view2_model_layer = build_mlp_net(layer_sizes2,  reg_par, view2_input, dropout=dropout)


    merge_layer = Concatenate(name='merge_layer')([view1_model_layer, view2_model_layer])
    model = Model(inputs=[view1_input, view2_input], outputs=merge_layer)

    opt = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=opt)

    model.summary()

    return model


def build_mlp_net(layer_sizes, reg_par, view_input_layer, dropout=None):

    layer = view_input_layer

    print("layer_sizes", layer_sizes)
    for l_id, ls in enumerate(layer_sizes):

        if l_id == len(layer_sizes)-1:
            activation = 'linear'
        else:
            activation = 'sigmoid'

        if dropout and l_id == len(layer_sizes)-1:
            layer = Dropout()(layer)

        layer = Dense(ls, activation=activation, kernel_regularizer=l2(reg_par))(layer)
    return layer
