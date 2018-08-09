import sys
sys.path.append(".")
import numpy as np
from data.datasets import dataloader
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

data = dataloader("CUB1")
print(data.keys())
classes = 200

features = data['features'].T
labels = data['labels'] - 1

train_loc = data['train_loc']
x_train = np.array([features[i] for i in np.reshape(train_loc,(train_loc.shape[0]))])
y_train_tmp = np.array([labels[i] for i in np.reshape(train_loc,(train_loc.shape[0]))])
y_train = np.eye(classes)[np.reshape(y_train_tmp,(train_loc.shape[0]))]

trainval_loc = data['trainval_loc']
x_trainval = np.array([features[i] for i in np.squeeze(trainval_loc)])
y_trainval = np.array([labels[i] for i in np.squeeze(trainval_loc)])
y_trainval = np.eye(classes)[np.squeeze(y_trainval)]

val_loc = data['val_loc']
x_val = np.array([features[i] for i in np.squeeze(val_loc)])
y_val = np.array([labels[i] for i in np.squeeze(val_loc)])
y_val = np.eye(classes)[np.squeeze(y_val)]

test_seen_loc = data['test_seen_loc']
x_test = np.array([features[i] for i in np.squeeze(test_seen_loc)])
y_test = np.array([labels[i] for i in np.squeeze(test_seen_loc)])
y_test = np.eye(classes)[np.squeeze(y_test)]


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(1024, activation='relu', input_dim=2048))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=300, batch_size=128,
                    validation_data=(x_trainval, y_trainval),
                    verbose=2, shuffle=False, callbacks=[early_stopping])

score = model.evaluate(x_val, y_val, batch_size=128)
print("val : ", score)
score = model.evaluate(x_test, y_test, batch_size=128)
print("test : ", score)
model.save('model.h5')

