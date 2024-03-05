import os.path
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


DATA_PATH = os.path.join('MP_Data')

actions = np.array(['Hello','Thanks','ILoveYou'])


number_of_sequences = 30
sequences_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences,labels = [], []

for action in actions:
    for seq in range(1,31):
        window = []
        for frame_num in range(1,sequences_length+1):
            res = np.load(os.path.join(DATA_PATH,action,str(seq), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Building and training LSTM Neural Network
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(32, 'relu'))
model.add(Dense(units=actions.shape[0], activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit(X_train,y_train,epochs=2000,callbacks=[tb_callback])
model.save("ISLR.keras")
model.summary()

