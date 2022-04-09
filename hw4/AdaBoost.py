import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(file):
    data_x = []
    data_y = []
    df = pd.read_csv(file)
    data_x = np.array(df.iloc[:, : 2])
    data_y = np.array(df.iloc[:, 2])
    return data_x, data_y


f = None
description = []
boundary = 0
y_pred = None
accuracy = 0


def ds_fit(train_x, train_y, weights=None):
    if weights is None:
        weights = np.ones_like(train_y)
    h = [(x[0], y) for (x, y) in zip(train_x, train_y)]
    v = [(x[1], y) for (x, y) in zip(train_x, train_y)]
    # Compute horizontal boundary
    lowest_err = len(train_y)
    for i in range(len(h) + 1):
        # Horizontal
        bound = h[i][0] if i < len(h) else float("inf")
        f1 = lambda x: 1 if ((x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)) else 0
        f2 = lambda x: 1 if ((x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)) else 0
        err1 = np.sum(weights * np.array(list(map(f1, h))))
        err2 = np.sum(weights * np.array(list(map(f2, h))))
        if err1 < lowest_err:
            lowest_err = err1
            description = "vertical <{} -1".format(
                boundary)  # horizontal axis < bound is classified as -1

            description = ['vert', boundary]

            f = lambda x: (-1 if x[0] < boundary else 1)
        if err2 < lowest_err:
            lowest_err = err2
            description = "vertical <{} 1".format(boundary)
            description = ['vert', boundary]
            f = lambda x: (1 if x[0] < boundary else -1)
        # Vertical
        bound = v[i][0] if i < len(v) else float("inf")
        f1 = lambda x: 1 if ((x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)) else 0
        f2 = lambda x: 1 if ((x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)) else 0
        err1 = np.sum(weights * np.array(list(map(f1, v))))
        err2 = np.sum(weights * np.array(list(map(f2, v))))
        if err1 < lowest_err:
            lowest_err = err1
            description = "horizontal <{} -1".format(boundary)
            description = ['hor', boundary]
            f = lambda x: (-1 if x[1] < boundary else 1)
        if err2 < lowest_err:
            lowest_err = err2
            description = "horizontal <{} 1".format(boundary)
            description = ['hor', boundary]
            f = lambda x: (1 if x[1] < boundary else -1)
    accuracy = 1 - lowest_err / len(train_y)
    return f


def ds_predict(f, test_x):
    return np.array(list(map(f, test_x)))


def compute_error(y, y_pred):
    return np.sum(np.abs(y + y_pred)) / (2 * y.shape[0])


class DecisionStump:
    """
    1-level decision tree, weak classifier for the adaboost algorithm.
    """

    def __init__(self):
        self.f = None
        self.description = None
        self.boundary = 0
        self.y_pred = None
        self.accuracy = 0

    def fit(self, train_x, train_y, weights=None):
        if weights is None:
            weights = np.ones_like(train_y)
        h = [(x[0], y) for (x, y) in zip(train_x, train_y)]
        v = [(x[1], y) for (x, y) in zip(train_x, train_y)]
        # Compute horizontal boundary
        lowest_err = len(train_y)
        for i in range(len(h) + 1):
            # Horizontal
            bound = h[i][0] if i < len(h) else float("inf")
            f1 = lambda x: 1 if ((x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)) else 0
            f2 = lambda x: 1 if ((x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)) else 0
            err1 = np.sum(weights * np.array(list(map(f1, h))))
            err2 = np.sum(weights * np.array(list(map(f2, h))))
            if err1 < lowest_err:
                lowest_err = err1
                self.boundary = bound
                self.description = "vertical <{} -1".format(
                    self.boundary)  # horizontal axis < bound is classified as -1
                self.description = ['vert', self.boundary]
                self.f = lambda x: (-1 if x[0] < self.boundary else 1)
            if err2 < lowest_err:
                lowest_err = err2
                self.boundary = bound
                self.description = "vertical <{} 1".format(self.boundary)
                self.description = ['vert', self.boundary]
                self.f = lambda x: (1 if x[0] < self.boundary else -1)
            # Vertical
            bound = v[i][0] if i < len(v) else float("inf")
            f1 = lambda x: 1 if ((x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)) else 0
            f2 = lambda x: 1 if ((x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)) else 0
            err1 = np.sum(weights * np.array(list(map(f1, v))))
            err2 = np.sum(weights * np.array(list(map(f2, v))))
            if err1 < lowest_err:
                lowest_err = err1
                self.boundary = bound
                self.description = "horizontal <{} -1".format(self.boundary)
                self.description = ['hor', self.boundary]
                self.f = lambda x: (-1 if x[1] < self.boundary else 1)
            if err2 < lowest_err:
                lowest_err = err2
                self.description = "horizontal <{} 1".format(self.boundary)
                self.description = ['hor', self.boundary]
                self.boundary = bound
                self.f = lambda x: (1 if x[1] < self.boundary else -1)
        self.accuracy = 1 - lowest_err / len(train_y)
        return self.f

    def predict(self, test_x):
        a = self.f
        self.y_pred = np.array(list(map(self.f, test_x)))
        return self.y_pred

    def compute_error(self, y, y_pred):
        self.accuracy = np.sum(np.abs(y + y_pred)) / (2 * y.shape[0])
        return self.accuracy


iteration = 0
learners = []
weighted_error = []
votes = []
weights = None
y_pred = None
accuracy = 0


def ada_fit(train_x, train_y, iteration=1):
    weights = np.ones(train_y.shape[0])
    for k in range(iteration):
        f = ds_fit(train_x, train_y, weights)
        train_y_pred = ds_predict(train_x)
        err = np.sum(weights * np.abs(train_y - train_y_pred) / 2) / np.sum(weights)
        vote = np.log((1 - err) / err) / 2
        weights = weights * np.exp(-train_y * vote * train_y_pred)  # update weights
        votes.append(vote)
        learners.append(f)


def ada_predict(test_x):
    s = np.zeros(test_x.shape[0])
    for k in range(iteration):
        s += votes[k] * ds_predict(learners[k], test_x)
    return np.array(list(map(lambda x: -1 if x < 0 else 1, s / sum(votes))))


def plot_result(train_x, train_y, test_x, test_y, iteration=1):
    plt.figure()
    accuracy = []
    iteration = 0
    weights = np.ones(train_y.shape[0])
    for k in range(iteration):
        iteration = k + 1
        model = DecisionStump()
        model.fit(train_x, train_y, weights)
        train_y_pred = model.predict(train_x)
        err = np.sum(weights * np.abs(train_y - train_y_pred) / 2) / np.sum(weights)
        vote = np.log((1 - err) / err) / 2
        weights = weights * np.exp(-train_y * vote * train_y_pred)
        votes.append(vote)
        learners.append(model)
        test_y_pred = ada_predict(test_x)
        accuracy.append(compute_error(test_y, test_y_pred))
    plt.plot([i + 1 for i in range(iteration)], accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.title("Test accuracy verses number of iterations")
    plt.savefig("1.png")

    return accuracy


class AdaBoost:
    def __init__(self):
        self.iteration = 0
        self.learners = []
        self.weighted_error = []
        self.votes = []
        self.weights = None
        self.y_pred = None
        self.accuracy = 0

    def fit(self, train_x, train_y, iteration=1):
        self.iteration = iteration
        self.weights = np.ones(train_y.shape[0])
        for k in range(iteration):
            model = DecisionStump()
            model.fit(train_x, train_y, self.weights)
            train_y_pred = model.predict(train_x)
            err = np.sum(self.weights * np.abs(train_y - train_y_pred) / 2) / np.sum(self.weights)
            vote = np.log((1 - err) / err) / 2
            self.weights = self.weights * np.exp(-train_y * vote * train_y_pred)  # update weights
            self.votes.append(vote)
            self.learners.append(model)

    def predict(self, test_x):
        s = np.zeros(test_x.shape[0])
        for k in range(self.iteration):
            s += self.votes[k] * self.learners[k].predict(test_x)
        self.y_pred = np.array(list(map(lambda x: -1 if x < 0 else 1, s / sum(self.votes))))
        return self.y_pred

    def eval_model(self, y, y_pred):
        self.accuracy = np.sum(np.abs(y + y_pred)) / (2 * y.shape[0])
        return self.accuracy

    def plot_weights(self, model, train_y, train_x, weights):
        pos_x = train_x[train_y == 1]
        neg_x = train_x[train_y == -1]
        weights = weights ** 2 * 100
        pos_weights = weights[train_y == 1]
        neg_weights = weights[train_y == -1]
        model.description
        db = model.description[1]

        plt.figure(figsize=(6, 6))
        plt.scatter(neg_x[:, 0], neg_x[:, 1], s=neg_weights, color='red', marker='_')
        plt.scatter(pos_x[:, 0], pos_x[:, 1], s=pos_weights, color='blue', marker='+')

        plt.title('Weighted Data', fontsize=16)
        plt.legend(['-1', '+1'], fontsize=16)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.show()

        if 'vert' in model.description[0]:
            print('hi')
            plt.figure(figsize=(6, 6))
            plt.scatter(neg_x[:, 0], neg_x[:, 1], s=neg_weights, color='red', marker='_')
            plt.scatter(pos_x[:, 0], pos_x[:, 1], s=pos_weights, color='blue', marker='+')
            plt.plot([db, db], [-2, 2], color='black', label='_nolegend_')
            plt.fill([-2,-2, db, db], [-2, 2, 2, -2], color='red', alpha=0.1, label='_nolegend_')
            plt.fill([2, 2, db, db], [-2, 2, 2, -2], color='blue', alpha=0.1, label='_nolegend_')
        else:
            plt.figure(figsize=(6, 6))
            plt.scatter(neg_x[:, 0], neg_x[:, 1], s=neg_weights, color='red', marker='_')
            plt.scatter(pos_x[:, 0], pos_x[:, 1], s=pos_weights, color='blue', marker='+')
            plt.plot([-2, 2], [db, db], color='black', label='_nolegend_')
            plt.fill([-2,-2, 2, 2], [-2, db, db, -2], color='red', alpha=0.1, label='_nolegend_')
            plt.fill([-2,-2, 2, 2], [2, db, db, 2], color='blue', alpha=0.1, label='_nolegend_')

        plt.title('Weighted Data w/ Tree', fontsize=16)
        plt.legend(['-1', '+1'], fontsize=16)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.show()
        print('here')


    def plot_result(self, train_x, train_y, test_x, test_y, iteration=1):
        plt.figure()
        accuracy = []
        self.iteration = 0
        self.weights = np.ones(train_y.shape[0])
        for k in range(iteration):
            self.iteration = k + 1
            model = DecisionStump()
            model.fit(train_x, train_y, self.weights)
            train_y_pred = model.predict(train_x)
            err = np.sum(self.weights * np.abs(train_y - train_y_pred) / 2) / np.sum(self.weights)
            vote = np.log((1 - err) / err) / 2

            # 5.1
            self.plot_weights(model, train_y_pred, train_x, self.weights)


            self.weights = self.weights * np.exp(-train_y * vote * train_y_pred)  # update weights
            self.votes.append(vote)
            self.learners.append(model)
            test_y_pred = self.predict(test_x)
            self.eval_model(test_y, test_y_pred)
            accuracy.append(self.accuracy)
        plt.plot([i + 1 for i in range(iteration)], accuracy)
        plt.xlabel("Number of iterations")
        plt.ylabel("Accuracy")
        plt.title("Test accuracy verses number of iterations")
        plt.savefig("1.png")


if __name__ == "__main__":
    train_x, train_y = read_data("data_22/train_adaboost.csv")
    test_x, test_y = read_data("data_22/test_adaboost.csv")
    AB = AdaBoost()
    AB.plot_result(train_x, train_y, test_x, test_y, iteration=50)  # 5.1 plot
    # plot_result(train_x, train_y, test_x, test_y, iteration=50)  # 5.1 plot
    print(AB.accuracy)  # 5.2 test accuracy after 50 iterations
    print(AB.learners[0].description)  # 5.3
    print(AB.learners[1].description)
    print(AB.learners[2].description)
