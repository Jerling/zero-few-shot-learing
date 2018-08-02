import sys
import numpy as np
sys.path.append(".")
print(sys.path)
from data.datasets import dataloader
from scipy.linalg import solve_sylvester
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD

lambda_ = .2

def sae(x , s, lambda_):
    A = np.matmul(s, s.T)
    B = lambda_ * np.matmul(x, x.T)
    C = (1 + lambda_) * np.matmul(s, x.T)
    return solve_sylvester(A, B, C)

def s2f(W, x):
    return np.matmul(W, x.T)

def labeltoindex(y):
    dicty_tr_ = {}
    idx = 0
    list_y = []
    for data in y:
        data = data[0]
        if data not in dicty_tr_.keys():
            dicty_tr_[data] = idx
            idx += 1
        list_temp = np.zeros(50)
        list_temp[dicty_tr_[data]] = 1.0
        list_y.append(list_temp)
    return np.array(list_y)

def nlpway(x_train, y_train):
    y_train = labeltoindex(y_train)
    print('x_train :', x_train.shape)
    print('y_train :', y_train.shape)
    model =  Sequential()
    model.add(Dense(312, activation='relu', input_dim=312))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer = sgd,
                  metrics = ['accuracy'])
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=8)

    return model.evaluate(x_train[:500], y_train[:500],batch_size=8)

def main():
    # 加载数据集
    dataset = dataloader()
    dicl = list(dataset.keys())
    for dic in dicl:
        print(dic,': ', dataset[dic].shape)

    test_x = dataset[dicl[1]] # (2933, 1024)
    test_y = dataset[dicl[2]] # (2933, 1)

    x_train, x_test, y_train, y_test = train_test_split(test_x, test_y, test_size=0.3)

    # preprocessing
    # stadard
    scaler = preprocessing.StandardScaler().fit(dataset[dicl[-1]])
    train_att = scaler.transform(dataset[dicl[-1]])
    scaler = preprocessing.StandardScaler().fit(dataset[dicl[0]])
    test_att = scaler.transform(dataset[dicl[0]])

    # MinMaxScaler
    #  train_att = preprocessing.MinMaxScaler().fit_transform(dataset[dicl[-1]])
    #  test_att = preprocessing.MinMaxScaler().fit_transform(dataset[dicl[0]])

    # Nomalization
    #  train_att = preprocessing.normalize(dataset[dicl[-1]], norm="l1")
    #  test_att = preprocessing.normalize(dataset[dicl[0]], norm="l1")
    #  train_att = preprocessing.normalize(dataset[dicl[-1]], norm="l2")
    #  test_att = preprocessing.normalize(dataset[dicl[0]], norm="l2")

    # cumpute W 
    W = sae(dataset[dicl[-3]].T, train_att.T, lambda_)
    #  print(W.shape)

    # reconstruct s
    s_ = s2f(W, dataset[dicl[1]])
    print(s_.shape)
    
    # compute consine similarity between s_ and test_att
    dist = cosine(s_.T, test_att)
    #  print(dist.shape)

    # get the index of the most similarity label
    y_ = np.argmax(dist, axis=1)
    #  print(y_.shape)
    #  print(y_)

    # cumpute the accuracy of testSet
    print("The accuracy is : ",(np.equal(dataset[dicl[2]], dataset[dicl[-2]][y_])).mean())

    x_train = s2f(W, test_x)
    score = nlpway(x_train.T, test_y)
    print('Score : ', score)


if __name__ == '__main__':
    main()
