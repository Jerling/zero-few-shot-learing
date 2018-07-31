import sys
import numpy as np
sys.path.append(".")
print(sys.path)
from data.datapath import dataloader
from scipy.linalg import solve_sylvester
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity as cosine

lambda_ = .2

def sae(x , s, lambda_):
    A = np.matmul(s, s.T)
    B = lambda_ * np.matmul(x, x.T)
    C = (1 + lambda_) * np.matmul(s, x.T)
    return solve_sylvester(A, B, C)

def s2f(W, x):
    return np.matmul(W, x.T)

def main():
    # 加载数据集
    dataset = dataloader()
    dicl = list(dataset.keys())
    for dic in dicl:
        print(dic,': ', dataset[dic].shape)

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
    #  print(s_.shape)
    
    # compute consine similarity between s_ and test_att
    dist = cosine(s_.T, test_att)
    #  print(dist.shape)

    # get the index of the most similarity label
    y_ = np.argmax(dist, axis=1)
    #  print(y_.shape)
    #  print(y_)

    # cumpute the accuracy of testSet
    print("The accuracy is : ",(np.equal(dataset[dicl[2]], dataset[dicl[-2]][y_])).mean())


if __name__ == '__main__':
    main()
