"""
Artificial Neural Network
Author: Elvis Pan
Email: ypan2@andrew.cmu.edu
"""
from numba import jit, cuda
import csv
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

_EXP = 0  # large number to avoid exponentiation overflow


def read_data(filename):
    x = []
    y = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            x.append([float(d) for d in line[:-1]])
            vec = np.zeros(10)
            vec[int(float(line[-1]))] = 1
            y.append(vec)
        f.close()
    return np.array(x), np.array(y)


def read_params(filename):
    param = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data = [float(d) for d in line.strip('\n').split(',')]
            param.append(data)
        f.close()
    return np.array(param)


def save_model(Model, filename):
    with open(filename, "wb") as f:
        pickle.dump(Model, f, pickle.HIGHEST_PROTOCOL)


class ANN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, regularization=0):
        self.M = input_dim
        self.D = hidden_dim
        self.K = output_dim
        self.x = None
        self.a = None
        self.z = None
        self.b = None
        self.y = None
        self.y_pred = None
        self.loss = 0
        self.loss_list = []
        self.accuracy = 0
        self.accuracy_list = []
        self.alpha = np.random.standard_normal((self.D, self.M))
        # self.alpha_bias = np.random.standard_normal((self.D, 1))
        self.alpha_bias = [[1],[1],[1],[1]]
        self.beta = np.random.standard_normal((self.K, self.D))
        # self.beta_bias = np.random.standard_normal((self.K, 1))
        self.beta_bias = [[1],[1],[1]]
        self.learning_rate = learning_rate
        self.reg = regularization

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        return np.exp(x - _EXP) / np.sum(np.exp(x - _EXP))

    def xeloss(self, pred, labels):
        return -np.sum(labels * np.log(pred)) + self.reg * (np.sum(self.alpha ** 2) + np.sum(self.beta ** 2))
        # return -np.sum(labels * np.log(pred))
    def xeloss_mult(self, pred, labels):
        return -np.sum(labels * np.log(pred)) / len(pred)

    def forward(self, x, y):
        self.x = x.reshape((self.M, 1))
        self.y = y.reshape((self.K, 1))
        self.a = self.alpha_bias + np.matmul(self.alpha, self.x)
        # self.a = [[1],[1],[1],[1]] + np.matmul(self.alpha, self.x)
        self.z = self.sigmoid(self.a)
        self.b = self.beta_bias + np.matmul(self.beta, self.z)
        # self.b = [[1],[1],[1]] + np.matmul(self.beta, self.z)

        self.y_pred = self.softmax(self.b)
        self.loss = self.xeloss(self.y_pred, self.y)

    def backward(self):
        b_grad = self.y_pred - self.y
        beta_grad = np.matmul(b_grad, self.z.T) + 2 * self.reg * self.beta
        z_grad = np.matmul(self.beta.T, b_grad)
        a_grad = z_grad * self.z * (1 - self.z)
        alpha_grad = np.matmul(a_grad, self.x.T) + 2 * self.reg * self.alpha
        self.beta -= self.learning_rate * beta_grad
        self.beta_bias -= self.learning_rate * b_grad
        self.alpha -= self.learning_rate * alpha_grad
        self.alpha_bias -= self.learning_rate * a_grad

    def fit(self, train_x, train_y, epoch=1):
        for e in range(epoch):
            for i in range(len(train_x)):
                x = train_x[i]
                y = train_y[i]
                self.forward(x, y)
                self.backward()
            self.accuracy = self.compare(self.predict(train_x), train_y)
            self.accuracy_list.append(self.accuracy)
            self.loss_list.append(self.xeloss_mult(self.predict(train_x), train_y))

    def predict(self, test_x):
        res = []
        for x in test_x:
            x_mat = x.reshape((self.M, 1))
            a = self.alpha_bias + np.matmul(self.alpha, x_mat)
            z = self.sigmoid(a)
            b = self.beta_bias + np.matmul(self.beta, z)
            y_pred = self.softmax(b)
            res.append([x[0] for x in y_pred])
        return np.array(res)

    @staticmethod
    def compare(y_pred, y):
        cnt = 0
        y1 = np.argmax(y, axis=1)
        y2 = np.argmax(y_pred, axis=1)
        for i in range(len(y)):
            if y1[i] == y2[i]:
                cnt += 1
        return cnt / len(y)

    def plot_loss(self, epoch, train_x, train_y, test_x, test_y, filename):
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        for e in range(epoch):
            loss_total = 0
            for i in range(len(train_x)):
                x = train_x[i]
                y = train_y[i]
                self.forward(x, y)
                self.backward()
                loss_total += self.loss
            train_pred = self.predict(train_x)
            test_pred = self.predict(test_x)
            self.loss_list.append(self.xeloss_mult(train_pred, train_y))
            train_loss.append(self.xeloss_mult(train_pred, train_y))
            test_loss.append(self.xeloss_mult(test_pred, test_y))
            self.accuracy_list.append(self.compare(test_pred, test_y))
            train_accuracy.append(self.compare(train_pred, train_y))
            test_accuracy.append(self.compare(test_pred, test_y))
            print("Epoch {} finished".format(e + 1))
        plt.figure()
        plt.plot([i + 1 for i in range(epoch)], train_loss, label="Training Loss")
        plt.plot([i + 1 for i in range(epoch)], test_loss, label="Test Loss")
        plt.legend()
        plt.xlabel("Number of Epoch")
        plt.ylabel("Average Loss")
        plt.title("Average Training Loss vs Test Loss")
        plt.savefig(filename)
        return train_loss, test_loss, train_accuracy, test_accuracy


if __name__ == "__main__":
    # Load dataset
    train_x, train_y = read_data("data/smallTrain.csv")
    test_x, test_y = read_data("data/smallValidation.csv")

    def new_model():
        # Initialize model
        Model = ANN(784, 256, 10, learning_rate=0.01)

        # M = ANN(M.M, M.D, M.K, learning_rate=M.learning_rate)
        Model.alpha = read_params("params/alpha1.txt")
        Model.alpha_bias = read_params("params/beta1.txt")
        Model.beta = read_params("params/alpha2.txt")
        Model.beta_bias = read_params("params/beta2.txt")
        return Model

    Model = new_model()
    Model.fit([train_x[0]], [train_y[0]], epoch=1)
    print(Model.a[9])  # Q5.1, zero indexed so a[9] = a_{10}
    print(Model.z[19])  # Q5,2, zero indexed so z[19] = z_{20}
    print(Model.y_pred)  # Q5.3
    print("1 epoch finished")

    Model = new_model()
    Model.fit(train_x, train_y, epoch=3)
    print(Model.beta_bias)  # Q5.4
    print("3 epoch finished")

    # Time for breakpoint
    Model = new_model()
    Model.fit(train_x, train_y, epoch=15)
    print(Model.loss_list)  # Q5.5
    print(Model.accuracy_list)  # Q5.6
    print("15 epoch finished")

    # 100 epochs
    Model = new_model()
    Model.plot_loss(100, train_x, train_y, test_x, test_y, "result/100.png")  # Q5.7
    print(Model.xeloss_mult(Model.predict(train_x), train_y))  # Q5.8.1
    print(Model.compare(Model.predict(test_x), test_y))  # Q5.8.2

    # Q5.9, three plots
    Model = ANN(784, 256, 10, learning_rate=0.01)
    Model.alpha = np.zeros((Model.D, Model.M))
    Model.alpha_bias = np.zeros((Model.D, 1))
    Model.beta = np.zeros((Model.K, Model.D))
    Model.beta_bias = np.zeros((Model.K, 1))
    Model.plot_loss(100, train_x, train_y, test_x, test_y, "result/zero.png")

    Model = ANN(784, 256, 10, learning_rate=0.01)
    Model.alpha = np.random.uniform(0, 1, (Model.D, Model.M)) / (Model.D * (Model.M + 1))
    Model.alpha_bias = np.random.uniform(0, 1, (Model.D, 1)) / (Model.D * (Model.M + 1))
    Model.beta = np.random.uniform(0, 1, (Model.K, Model.D)) / (Model.K * (Model.D + 1))
    Model.beta_bias = np.random.uniform(0, 1, (Model.K, 1)) / (Model.K * (Model.D + 1))
    Model.plot_loss(100, train_x, train_y, test_x, test_y, "result/uniform.png")

    Model = ANN(784, 256, 10, learning_rate=0.01)
    Model.alpha = np.random.standard_normal((Model.D, Model.M))
    Model.alpha_bias = np.random.standard_normal((Model.D, 1))
    Model.beta = np.random.standard_normal((Model.K, Model.D))
    Model.beta_bias = np.random.standard_normal((Model.K, 1))
    Model.plot_loss(100, train_x, train_y, test_x, test_y, "result/normal.png")