from sklearn import datasets
import numpy as np

EPOCH = 10000
LEARNING_RATE = 0.1


class LogisticRegression:
    def __init__(self, data_X):
        self.theta = np.zeros(data_X.shape[1] + 1)

    def fit(self, data_X: np.ndarray, data_Y, eta, threshold=0.5):
        data_X = np.concatenate((np.ones((data_X.shape[0], 1)), data_X), axis=1)

        for epoch in range(EPOCH):
            z = np.dot(data_X, self.theta)
            h = sigmoid(z)
            print(f'Epoch {epoch}. Cross entropy loss: {cross_entropy(h, data_Y)}, '
                  f'Accuracy: {self.accuracy(h, data_Y, threshold)}')
            gradient = np.dot(data_X.T, (h - data_Y)) / data_Y.size
            self.theta -= eta * gradient

    def predict(self, data, threshold=0.5):
        data_c = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
        return sigmoid(np.dot(data_c, self.theta)) >= threshold

    def accuracy(self, h, data_Y, threshold):
        correctness = [1 if ((h[i] >= threshold and data_Y[i] == 1) or (h[i] < threshold and data_Y[i] == 0)) else 0 for
                       i in range(h.shape[0])]
        return np.count_nonzero(correctness) / h.shape[0]


def sigmoid(x, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    else:
        return s


def cross_entropy(out, label):
    return (-label * np.log(out) - (1 - label) * np.log(1 - out)).mean()


def main():
    cancer_data = datasets.load_breast_cancer()
    X = cancer_data.data[:1000, :]
    X = X / X.max(axis=0)
    y = cancer_data.target
    model = LogisticRegression(X)
    model.fit(X, y, LEARNING_RATE)
    preds = model.predict(X)
    print(preds == y)


if __name__ == '__main__':
    main()
