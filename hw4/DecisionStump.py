import csv
import numpy as np


def read_data(filename):
    data_x = []
    data_y = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            data_x.append([float(line[0]), float(line[1])])
            data_y.append(int(line[-1]))
        f.close()
    return np.array(data_x), np.array(data_y)


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
                self.f = lambda x: (-1 if x[0] < self.boundary else 1)
            if err2 < lowest_err:
                lowest_err = err2
                self.boundary = bound
                self.description = "vertical <{} 1".format(self.boundary)
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
                self.f = lambda x: (-1 if x[1] < self.boundary else 1)
            if err2 < lowest_err:
                lowest_err = err2
                self.description = "horizontal <{} 1".format(self.boundary)
                self.boundary = bound
                self.f = lambda x: (1 if x[1] < self.boundary else -1)
        self.accuracy = 1 - lowest_err / len(train_y)
        return self.f

    def predict(self, test_x):
        self.y_pred = np.array(list(map(self.f, test_x)))
        return self.y_pred

    def compute_error(self, y, y_pred):
        self.accuracy = np.sum(np.abs(y + y_pred)) / (2 * y.shape[0])
        return self.accuracy


if __name__ == "__main__":
    train_x, train_y = read_data("datasets/train_adaboost.csv")
    test_x, test_y = read_data("datasets/test_adaboost.csv")
    DS = DecisionStump()
    DS.fit(train_x, train_y)
    print(DS.accuracy)
