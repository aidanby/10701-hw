import numpy as np
import matplotlib.pyplot as plt

_EXP = 0


def load_data_small():
    """ 
    Load small training and validation dataset

    Returns a tuple of length 4 with the following objects:
    X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
    y_train: An N_train-x-1 ndarray contraining the labels
    X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
    y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ 
    Load medium training and validation dataset

    Returns a tuple of length 4 with the following objects:
    X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
    y_train: An N_train-x-1 ndarray contraining the labels
    X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
    y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ 
    Load large training and validation dataset

    Returns a tuple of length 4 with the following objects:
    X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
    y_train: An N_train-x-1 ndarray contraining the labels
    X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
    y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    Arguments:
        - input: input vector (N, in_features + 1) 
            WITH bias feature added as 1st col
        - p: parameter matrix (out_features, in_features + 1)
            WITH bias parameter added as 1st col (i.e. alpha / beta in the writeup)

    Returns:
        - output vector (N, out_features)
    """
    # linear_out = np.dot(input, p[:, 1:]) + p[:, 0]
    linear_out = np.matmul(p, input)
    return linear_out


def sigmoidForward(a):
    """
    Arguments:
        - a: input vector (N, dim)

    Returns:
        - output vector (N, dim)
    """
    return 1 / (1 + np.exp(-a))


def softmaxForward(b):
    """
    Arguments:
        - b: input vector (N, dim)

    Returns:
        - output vector (N, dim)
    """
    return np.exp(b - _EXP) / np.sum(np.exp(b - _EXP))


def crossEntropyForward(hot_y, y_hat):
    """
    Arguments:
        - hot_y: 1-hot encoding for true labels (N, K), where K is the # of classes
        - y_hat: (N, K) vector of probabilistic distribution for predicted label

    Returns:
        - cross entropy loss (scalar)
    """
    N, K = np.shape(y_hat)

    return -np.sum(hot_y * np.log(y_hat)) / N

    pass


def NNForward(x, y, alpha, beta):
    """
    Arguments:
        - x: input vector (N, M+1)
            WITH bias feature added as 1st col
        - y: ground truth labels (N,)
        - alpha: alpha parameter matrix (D, M+1)
            WITH bias parameter added as 1st col
        - beta: beta parameter matrix (K, D+1)
            WITH bias parameter added as 1st col

    Returns (refer to writeup for details):
        - a: 1st linear output (N, D)
        - z: sigmoid output WITH bias feature added as 1st col (N, D+1)
        - b: 2nd linear output (N, K)
        - y_hat: softmax output (N, K)
        - J: cross entropy loss (scalar)

    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    # TODO bias is confusing
    x = x[:, 1:]
    alpha = alpha[:, 1:]
    beta = beta[:, 1:]
    alpha_bias = alpha[:, 0]
    beta_bias = beta[:, 0]
    N,M = np.shape(x)
    K,D = np.shape(beta)
    x = x.reshape((M, 1))
    y = y.reshape((K, 1))
    a = alpha_bias + np.matmul(alpha, x)
    z = sigmoidForward(a)
    b = beta_bias + np.matmul(beta, z)
    y_hat = softmaxForward(b)

    J = -np.sum(y * np.log(y_hat)) / len(y_hat)
    return a,z,b,y_hat,J


def softmaxBackward(hot_y, y_hat):
    """
    Arguments:
        - hot_y: 1-hot encoding for true labels (N, K) where K is the # of classes
        - y_hat: (N, K) vector of probabilistic distribution for predicted label
    """
    #dl/dy_hat
    return y_hat - hot_y


def linearBackward(prev, p, grad_curr):
    """
    Arguments:
        - prev: previous layer WITH bias feature
        - p: parameter matrix (alpha/beta) WITH bias parameter
        - grad_curr: gradients for current layer

    Returns:
        - grad_param: gradients for parameter matrix (i.e. alpha / beta)
            This should have the same shape as the parameter matrix.
        - grad_prevl: gradients for previous layer

    TIP: Check your dimensions.
    """
    # dl/dz
    beta_grad = np.matmul(grad_curr, prev.T)
    z_grad = np.matmul(p.T, grad_curr)
    return beta_grad, z_grad


def sigmoidBackward(curr, grad_curr):
    """
    Arguments:
        - curr: current layer WITH bias feature
        - grad_curr: gradients for current layer

    Returns: 
        - grad_prevl: gradients for previous layer
    """
    #dl/da
    a_grad = grad_curr * curr * (1 - curr)
    return a_grad


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    Arguments:
        - x: input vector (N, M)
        - y: ground truth labels (N,)
        - alpha: alpha parameter matrix (D, M+1)
            WITH bias parameter added as 1st col
        - beta: beta parameter matrix (K, D+1)
            WITH bias parameter added as 1st col
        - z: z as per writeup
        - y_hat: (N, K) vector of probabilistic distribution for predicted label

    Returns:
        - g_alpha: gradients for alpha
        - g_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    g_b = softmaxBackward(y, y_hat)
    g_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z, g_z)
    g_alpha, g_z = linearBackward(x, alpha, g_a)
    return g_alpha, g_beta, g_b, g_z, g_a



def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    Arguments:
        - tr_x: training data input (N_train, M)
        - tr_y: training labels (N_train, 1)
        - valid_x: validation data input (N_valid, M)
        - valid_y: validation labels (N_valid, 1)
        - hidden_units: Number of hidden units
        - num_epoch: Number of epochs
        - init_flag:
            - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
            - False: Initialize weights and bias to 0
        - learning_rate: Learning rate

    Returns:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    # if init_flag:
    #     alpha = np.random.standard_normal((D, M))


    alpha = np.array([[1, 2, -3, 0, 1, -3], [2, 1, 1, 1, 0, 2], [3, 2, 2, 2, 2, 1], [2, 0, 3, 1, -2, 2]],
                           dtype=float)
    beta = np.array([[1, 2, -2, 3], [2, -1, 3, 1], [3, 1, -1, 1]], dtype=float)

    tr_losses = []
    val_losses = []
    for e in range(num_epoch):
        for i in range(len(tr_x)):
            x = tr_x[i]
            y = tr_y[i]
            a,z,b,y_hat,J = NNForward(x, y, alpha, beta)
            g_alpha, g_beta, g_b, g_z, g_a = NNBackward(x, y, alpha, beta, z, y_hat)
            beta -= learning_rate * g_b
            alpha -= learning_rate * g_a
            tr_losses.append(J)

    for e in range(num_epoch):
        for i in range(len(valid_x)):
            x = valid_x[i]
            y = valid_y[i]
            a,z,b,y_hat,J = NNForward(x, y, alpha, beta)
            val_losses.append(J)

    return alpha, beta, tr_losses, val_losses


def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    Arguments:
        - tr_x: training data input (N_train, M)
        - tr_y: training labels (N_train, 1)
        - valid_x: validation data input (N_valid, M)
        - valid_y: validation labels (N-valid, 1)
        - tr_alpha: alpha weights WITH bias
        - tr_beta: beta weights WITH bias

    Returns:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    res = []
    for x in tr_x:
        x_mat = x.reshape((np.shape(tr_x), 1))
        a = np.matmul(tr_alpha, x_mat)
        z = sigmoidForward(a)
        b = np.matmul(tr_beta, z)
        y_pred = softmaxForward(b)
        res.append([x[0] for x in y_pred])
        #TODO

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ 
    Main function to train and validate your neural network implementation.
        
    Arguments:
        - X_train: training input in (N_train, M) array. Each value is binary, in {0,1}.
        - y_train: training labels in (N_train, 1) array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        - X_val: validation input in (N_val, M) array. Each value is binary, in {0,1}.
        - y_val: validation labels in (N_val, 1) array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        - num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        - num_hidden: Positive integer representing the number of hidden units.
        - init_flag: Boolean value of True/False
            - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
            - False: Initialize weights and bias to 0
        - learning_rate: Float value specifying the learning rate for SGD.

    Returns:
        - loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        - loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        - err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        - err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        - y_hat_train: A list of integers representing the predicted labels for training data
        - y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    err_train = None
    err_val = None
    y_hat_train = None
    y_hat_val = None

    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)
