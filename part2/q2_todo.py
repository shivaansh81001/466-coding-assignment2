# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels

def plot_graphs(train_losses,valid_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, MaxEpoch + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, MaxEpoch + 1), valid_accs, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here

    l=X@W
    l-=np.max(l,axis=1,keepdims=True) #subtracting the largest zi to account for calc
    y=np.exp(l)
    y_new=y/np.sum(y,axis=1,keepdims=True)
    t_hat=np.argmax(y_new,axis=1).reshape(-1,1)
    
    loss=None
    acc=None
    
    if t is not None:
        t_one_hot = np.eye(W.shape[1])[t.flatten()]
        loss=-np.sum(t_one_hot*np.log(y+1e-16))/X.shape[0]
        acc=np.mean(t_hat==t)
        

    return y, t_hat, loss, acc


#https://spotintelligence.com/2023/08/16/softmax-regression/

def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    dplus1_train=X_train.shape[1]
    #print(dplus1_train)
    
    N_val = X_val.shape[0]

    # TODO Your code here
    classes=10
    W=np.random.randn(dplus1_train,classes)*0.01
    #print(y_train)
    train_losses = []
    valid_accs = []
    acc_best = 0
    epoch_best = 0
    W_best = None
    
    for epoch in range(MaxEpoch):
        #ChatGPT prompt "Shuffle this data for me for epochs"
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        t_train_shuffled = t_train[indices]
        
        for batch in range(0,len(X_train),batch_size):
            end = batch+batch_size
            X_batch = X_train_shuffled[batch:end]
            t_batch = t_train_shuffled[batch:end]
            #ChatGPT prompt "Shuffle this data for me for epochs"
            
            l=X_batch@W
            l_new=l-np.max(l,axis=1,keepdims=True)
            y=np.exp(l_new)/np.sum(np.exp(l_new),axis=1,keepdims=True)
            #ChatGPT prompt "how to implement one hot in softmax in python"
            t_one_hot=np.eye(classes)[t_batch.flatten()]
            
            gradient=(X_batch.T@(y-t_one_hot))/batch_size
            W-=alpha*gradient
            
            
        _,_,train_loss,_=predict(X_train,W,t_train)
        train_losses.append(train_loss)    
        
        _, _, _,val_acc=predict(X_val, W, t_val)
        if val_acc>acc_best:
            acc_best=val_acc
            epoch_best=epoch
            W_best=W.copy()
        
            
        valid_accs.append(val_acc)
        
        #print(f"Epoch {epoch+1} Accuracy = {val_acc*100:.2f} %")
        
    return epoch_best, acc_best,  W_best, train_losses, valid_accs


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


#print(X_train.shape, t_train.shape, X_val.shape,t_val.shape, X_test.shape, t_test.shape)
      
N_class = 10
alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

plot_graphs(train_losses, valid_accs)

print("Best Epoch: ",epoch_best)
print("Best validation Accuracy: ",round(acc_best*100,2)," %")

_, _, _, acc_test = predict(X_test, W_best, t_test)
print("Test Accuracy: ",round(acc_test*100,2)," %")
