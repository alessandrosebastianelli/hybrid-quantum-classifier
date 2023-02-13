from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from config import *

class LeNet5:
    """
        Keras implementation of the LeNet-5
    """

    def __init__(self, shape):
        self.shape = shape
        self.model = self.__build()
        self.model.compile(optimizer=Adam(CNN_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    @staticmethod
    def __build(self):
        x_in = Input(shape=self.shape)
        x = ZeroPadding2D(padding=(2, 2))(x_in)
        x = Conv2D(6, kernel_size=(5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(16, kernel_size=(5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(84, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
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

        self.model.save(CNN_MODEL_SAVE_PATH1)

        return history
