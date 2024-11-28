from utils import plot_data, generate_data
import numpy as np


"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    
    """
    
    m,n=X.shape
    w=np.zeros(n)
    b=0
    
    for i in range(1000):
        t_hat=1/(1+np.exp(-(np.dot(X,w)+b)))
        dw=(1/m)*np.dot(X.T,(t_hat-t))
        db=(1/m)*np.sum(t_hat-t)
        w-=0.1*dw
        b-=0.1*db
    #print(w,b)
    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    temp=1/(1+np.exp(-(X@w+b)))
    t=(temp>=0.5).astype(int)
    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    #print(X,t)
    temp=np.linalg.inv(X.T@X)@X.T@t
    #print(temp)
    w=temp
    b=temp[-1]
    #print(w)
    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    #print(X,w)
    temp=np.sum(X*w,axis=1)+b
    t=[1 if x>=0.5 else 0 for x in temp]
    #print(t)
    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    #print(t,t_hat)
    acc=np.mean(t==t_hat)*100
    #print("accuracy",acc)
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
