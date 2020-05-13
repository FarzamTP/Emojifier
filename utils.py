import numpy as np


def one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(X, Y, W, b, word_to_vec_map):
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):
        avg = sentence_to_avg(X[j], word_to_vec_map)
        
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    return pred