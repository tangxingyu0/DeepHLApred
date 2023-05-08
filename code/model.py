import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense,Flatten, Dropout, Bidirectional, LSTM,Conv1D,MaxPool1D,Activation,BatchNormalization
import tensorflow as tf



def getmodel():
    input = Input((45,))
    embedding = Embedding(input_dim=20, output_dim=45, input_length=45)(input)
    model_1 = Conv1D(filters=32, kernel_size=10, activation='relu', padding='same')(embedding)
    model_1 = MaxPool1D(pool_size=2)(model_1)
    model_1 = Activation('relu')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Dropout(0.5)(model_1)
    model_1 = Bidirectional(LSTM(16, return_sequences=True))(model_1)
    model_1 = Flatten()(model_1)

    model_2 = Conv1D(filters=32, kernel_size=8, activation='relu', padding='same')(embedding)
    model_2 = MaxPool1D(pool_size=2)(model_2)
    model_2 = Activation('relu')(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Dropout(0.5)(model_2)
    model_2 = Bidirectional(LSTM(16, return_sequences=True))(model_2)
    model_2 = Flatten()(model_2)

    model = tf.concat([model_1, model_2], axis=-1)
    model = Dense(16, activation='relu')(model)
    output = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     model.summary()
    return model

