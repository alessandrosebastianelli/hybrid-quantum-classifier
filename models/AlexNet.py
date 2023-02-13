from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from config import *

class AlexNet:
    """
        Keras implementation of the AlexNet
    """

    def __init__(self, shape):
        self.shape = shape
        self.model = self.__build()
        self.model.compile(optimizer=Adam(CNN_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    @staticmethod
    def __build():
        x_in = Input(shape=shape)
        x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(x_in)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = ZeroPadding2D(padding=(2, 2))(x)
        x = Conv2D(256, kernel_size=(5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(384, kernel_size=(3, 3), activation='relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(384, kernel_size=(3, 3), activation='relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=x_in, outputs=x)

        return model

    def train(self, train_loader, val_loader, steps_per_epoch, validation_steps):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None,
                           restore_best_weights=False)

        history = self.model.fit(
            train_loader,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_loader,
            validation_steps=validation_steps,
            epochs=CNN_EPOCHS,
            callbacks=[es],
        )

        self.model.save(CNN_MODEL_SAVE_PATH2)

        return history
