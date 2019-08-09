"""
Utility functions for Neural network
"""
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, BatchNormalization, Input, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam


def plot_history(history, label, loss_str='sc_outp_mean_absolute_error'):
    import matplotlib.pyplot as plt
    plt.plot(history.history[loss_str])
    plt.plot(history.history[f'val_{loss_str}'])
    plt.title(f'Loss for {label}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _ = plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def create_nn_model(input_shape):
    inp = Input(shape=(input_shape, ))

    x = Dense(256)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)

    #     x = Dense(1024)(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(alpha=0.05)(x)
    #     x = Dropout(0.2)(x)

    #     x = Dense(1024)(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(alpha=0.05)(x)
    #     x = Dropout(0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)

    #     x = Dense(512)(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(alpha=0.05)(x)
    #     x = Dropout(0.4)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)

    out1 = Dense(20, activation="linear", name='int_outp')(x)  #2 mulliken charge, tensor 6, tensor 12(others)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)

    #     x = Dense(128)(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(alpha=0.05)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="linear", name='sc_outp')(x)  #scalar_coupling_constant

    model = Model(inputs=inp, outputs=[out, out1])
    #     model = Model(inputs=inp, outputs=[out])
    return model


def train_nn(nn_config, train_X, train_Y, val_X, val_Y, test_X):
    model_name_wrt = f'molecule_model_{nn_config["type_enc"]}.hdf5'
    assert isinstance(nn_config['load_model'], bool)

    if nn_config['load_model'] is False:
        model = create_nn_model(train_X.shape[1])
        model.compile(loss='mse', metrics=['mae'], optimizer=Adam(lr=nn_config['lr']))

        # tensorboard_callback = TensorBoard("logs/" + datetime.now().strftime('%H:%M:%S'), update_freq='epoch')
        val_loss = 'val_sc_outp_mean_absolute_error'
        es = EarlyStopping(monitor=val_loss, mode='min', patience=30, verbose=0)
        rlr = ReduceLROnPlateau(monitor=val_loss, factor=0.1, patience=25, min_lr=1e-6, mode='auto', verbose=1)

        sv_mod = ModelCheckpoint(
            model_name_wrt, monitor='val_sc_outp_mean_absolute_error', save_best_only=True, period=1)
        train_Y = train_Y.values
        val_Y = val_Y.values
        history = model.fit(
            train_X, [train_Y[:, 0], train_Y[:, 1:]],
            validation_data=(val_X, [val_Y[:, 0], val_Y[:, 1:]]),
            epochs=nn_config['epochs'],
            verbose=0,
            batch_size=nn_config['batch_size'],
            callbacks=[es, rlr, sv_mod])

        plot_history(history, nn_config['type_enc'])
    else:
        print('Loading from file', model_name_wrt)

    model = load_model(model_name_wrt)
    output_dict = {
        'model': model,
        'train_prediction': model.predict(train_X)[0][:, 0],
        'val_prediction': model.predict(val_X)[0][:, 0],
        'test_prediction': model.predict(test_X)[0][:, 0],
    }
    return output_dict


def get_intermediate_Ydf(mulliken_df, magnetic_shielding_tensors_df, raw_train_df):
    interm_Y_atomdata_df = pd.merge(
        mulliken_df, magnetic_shielding_tensors_df, how='outer', on=['molecule_name', 'atom_index'])
    Y_cols = interm_Y_atomdata_df.columns.tolist()
    Y_cols.remove('molecule_name')
    Y_cols.remove('atom_index')

    interm_Y_df = raw_train_df[['molecule_name', 'atom_index_0', 'atom_index_1']].reset_index()
    interm_Y_df = pd.merge(
        interm_Y_df,
        interm_Y_atomdata_df,
        how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'])
    interm_Y_df.rename({c: f'{c}_0' for c in Y_cols}, axis=1, inplace=True)
    interm_Y_df.drop('atom_index', axis=1, inplace=True)

    interm_Y_df = pd.merge(
        interm_Y_df,
        interm_Y_atomdata_df,
        how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'])
    interm_Y_df.rename({c: f'{c}_1' for c in Y_cols}, axis=1, inplace=True)
    interm_Y_df.drop(['atom_index', 'atom_index_0', 'atom_index_1', 'molecule_name'], axis=1, inplace=True)
    interm_Y_df.set_index('id', inplace=True)

    # Normalization
    interm_Y_df = pd.DataFrame(
        StandardScaler().fit_transform(interm_Y_df), columns=interm_Y_df.columns, index=interm_Y_df.index)
    return interm_Y_df
