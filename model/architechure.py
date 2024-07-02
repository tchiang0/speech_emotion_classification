import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation


class ModelArchitechure():
    def __init__(self):
        pass
    
    def model_architechure_one(self, x_train, y_train):
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu',
                         input_shape=(x_train.shape[1], 1)))
        self.model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(y_train.shape[1], activation="softmax"))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return self.model

    def model_architechure_two(self, x_train, y_train):
        self.model = Sequential()
        self.model.add(Conv1D(256, 8, padding='same', input_shape=(x_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(256, 8, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling1D(pool_size=(8)))
        self.model.add(Conv1D(128, 8, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(128, 8, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(128, 8, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(128, 8, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling1D(pool_size=(8)))
        self.model.add(Conv1D(64, 8, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(64, 8, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(y_train.shape[1])) # Target class number
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        return self.model


class ModelFitEval():
    def __init__(self):
        pass
    
    def model_fit(self, model, x_train, y_train, x_test, y_test):
        history = model.fit(x_train, y_train, batch_size=64, epochs=150, validation_data=(x_test, y_test))
        return history

    def model_eval(self, history, x_test, y_test):
        print("Accuracy of model on test data: ", self.model.evaluate(x_test, y_test)[1]*100, "%")
        epochs = [i for i in range(150)]
        fig, ax = plt.subplots(1, 2)
        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        test_acc = history.history['val_accuracy']
        test_loss = history.history['val_loss']

        fig.set_size_inches(20,6)
        ax[0].plot(epochs , train_loss , label = 'Training Loss')
        ax[0].plot(epochs , test_loss , label = 'Testing Loss')
        ax[0].set_title('Training & Testing Loss')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")

        ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
        ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
        ax[1].set_title('Training & Testing Accuracy')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")
        plt.show()
