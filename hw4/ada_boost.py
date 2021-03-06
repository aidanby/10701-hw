import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(file):
    df = pd.read_csv(file)
    data_x = np.array(df.iloc[:, : 2])
    data_y = np.array(df.iloc[:, 2])
    return data_x, data_y


description = []
boundary = 0
iteration = 0
weighted_error = []
votes = []
weights = None
y_pred = None
accuracy = 0


def ada_fit(train_x, train_y, iteration=1):
    weights = np.ones(train_y.shape[0])
    for k in range(iteration):
        if weights is None:
            weights = np.ones_like(train_y)
        h = [(x[0], y) for (x, y) in zip(train_x, train_y)]
        v = [(x[1], y) for (x, y) in zip(train_x, train_y)]
        lowest_err = len(train_y)
        for i in range(len(h) + 1):
            bound = h[i][0] if i < len(h) else float("inf")
            f1 = lambda x: 1 if ((x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)) else 0
            f2 = lambda x: 1 if ((x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)) else 0
            err1 = np.sum(weights * np.array(list(map(f1, h))))
            err2 = np.sum(weights * np.array(list(map(f2, h))))
            if err1 < lowest_err:
                lowest_err = err1
                description = ['vert', boundary]
                f = lambda x: (-1 if x[0] < boundary else 1)
            if err2 < lowest_err:
                lowest_err = err2
                description = ['vert', boundary]
                f = lambda x: (1 if x[0] < boundary else -1)
            bound = v[i][0] if i < len(v) else float("inf")
            f1 = lambda x: 1 if ((x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)) else 0
            f2 = lambda x: 1 if ((x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)) else 0
            err1 = np.sum(weights * np.array(list(map(f1, v))))
            err2 = np.sum(weights * np.array(list(map(f2, v))))
            if err1 < lowest_err:
                lowest_err = err1
                description = ['hor', boundary]
                f = lambda x: (-1 if x[1] < boundary else 1)
            if err2 < lowest_err:
                lowest_err = err2
                description = ['hor', boundary]
                f = lambda x: (1 if x[1] < boundary else -1)
        train_y_pred = np.array(list(map(f, test_x)))
        err = np.sum(weights * np.abs(train_y - train_y_pred) / 2) / np.sum(weights)
        vote = np.log((1 - err) / err) / 2
        weights = weights * np.exp(-train_y * vote * train_y_pred)  # update weights
        votes.append(vote)
        learners.append(f)

        return vote, f, weights, err


def ada_predict(test_x):
    s = np.zeros(test_x.shape[0])
    for k in range(iteration):
        s += votes[k] * np.array(list(map(f, test_x)))
    return np.array(list(map(lambda x: -1 if x < 0 else 1, s / sum(votes))))


def plot_weights(model, train_y, train_x, weights):
    # 5.1 plots
    pos_x = train_x[train_y == 1]
    neg_x = train_x[train_y == -1]
    weights = weights ** 2 * 100
    pos_weights = weights[train_y == 1]
    neg_weights = weights[train_y == -1]
    db = description[1]

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
        plt.fill([-2, -2, db, db], [-2, 2, 2, -2], color='red', alpha=0.1, label='_nolegend_')
        plt.fill([2, 2, db, db], [-2, 2, 2, -2], color='blue', alpha=0.1, label='_nolegend_')
    else:
        plt.figure(figsize=(6, 6))
        plt.scatter(neg_x[:, 0], neg_x[:, 1], s=neg_weights, color='red', marker='_')
        plt.scatter(pos_x[:, 0], pos_x[:, 1], s=pos_weights, color='blue', marker='+')
        plt.plot([-2, 2], [db, db], color='black', label='_nolegend_')
        plt.fill([-2, -2, 2, 2], [-2, db, db, -2], color='red', alpha=0.1, label='_nolegend_')
        plt.fill([-2, -2, 2, 2], [2, db, db, 2], color='blue', alpha=0.1, label='_nolegend_')

    plt.title('Weighted Data w/ Tree', fontsize=16)
    plt.legend(['-1', '+1'], fontsize=16)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    plt.show()
    print('here')


def plot_acc(train_x, train_y, test_x, test_y, iteration=1):
    # 5.2 plots
    plt.figure()
    accuracy = []
    iteration = 0
    weights = np.ones(train_y.shape[0])
    for k in range(iteration):
        iteration = k + 1
        vote, f, weights, err = ada_fit(train_x, train_y, weights)
        train_y_pred = ada_predict(train_x)
        err = np.sum(weights * np.abs(train_y - train_y_pred) / 2) / np.sum(weights)
        vote = np.log((1 - err) / err) / 2
        weights = weights * np.exp(-train_y * vote * train_y_pred)
        plot_weights(f, train_y, train_x, weights)
        votes.append(vote)
        test_y_pred = ada_predict(test_x)
        accuracy.append(np.sum(np.abs(y + y_pred)) / (2 * y.shape[0]))
    plt.plot([i + 1 for i in range(iteration)], accuracy)
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.title("Test accuracy verses number of iterations")
    plt.savefig("q5_1.png")
    return accuracy


def pca_analysis():
    # 4.2 matrix calculations
    x = [[-2, -2, -2, 0, 0], [-2, -2, -2, 0, 0], [4, 4, 4, 0, 0], [0, 0, 0, -2, -2], [0, 0, 0, 2, 2], [0, 0, 0, 0, 0]]
    import sklearn
    from sklearn.decomposition import PCA
    y = np.linalg.svd(x)
    print(y[0])
    print(y[1])
    print(y[2])
    pca = PCA(n_components=1)
    a = pca.fit_transform(x)
    print(np.var(a))
    print(np.mean((x-pca.inverse_transform(a))**2))


if __name__ == "__main__":

    # train_x, train_y = read_data("data_22/train_adaboost.csv")
    # test_x, test_y = read_data("data_22/test_adaboost.csv")
    # plot_acc(train_x, train_y, test_x, test_y, iteration=50)

    pca_analysis()

