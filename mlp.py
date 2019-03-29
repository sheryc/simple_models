import gzip
import numpy as np
from numba import jit

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
BATCH_SIZE = 1000
EPOCH = 100
VALIDATION_SIZE = 5000
LAYERS = [IMAGE_SIZE * IMAGE_SIZE, 50, 20, 10]
LEARNING_RATE = 0.1
L2_NORM = 0.0000001


class MLP(object):
    def __init__(self, layers: list):
        self.layers = np.array(layers)
        self.num_layers = len(layers)
        self.weights = [np.random.randn(layers[l], layers[l - 1]) / np.sqrt(layers[l - 1]) * 0.01 for l in
                        range(1, len(layers))]
        self.biases = [np.random.randn(layers[l], 1) * 0.01 for l in range(1, len(layers))]

    @jit
    def forward(self, data):
        data = np.array(data)
        z = data.reshape(IMAGE_SIZE * IMAGE_SIZE, 1)
        a_list = [data]
        z_list = [data]
        for l in range(0, len(self.layers) - 1):
            w, b = self.weights[l], self.biases[l]
            a = np.dot(w, z) + b
            if not l == len(self.layers) - 2:
                z = sigmoid(a)
            else:
                z = softmax(a)
            a_list.append(a)
            z_list.append(z)
        return z_list, a_list

    def backward(self, data_x, data_y):
        delta_w = [np.zeros_like(w) for w in self.weights]
        delta_b = [np.zeros_like(b) for b in self.biases]
        z_list, a_list = self.forward(data_x)
        delta_b[-1] = (z_list[-1] - data_y) * softmax(a_list[-1], derivative=True)
        delta_w[-1] = np.dot(delta_b[-1], z_list[-2].transpose())
        for l in range(len(self.layers) - 3, 0, -1):
            # delta_b_j = delta_j * \partial{z}/\partial{a} * 1
            # delta_w_j = delta_j * \partial{z}/\partial{a} * z[j-1]
            delta_b[l] = np.dot(self.weights[l + 1].transpose(), delta_b[l + 1]) * sigmoid(a_list[l + 1],
                                                                                           derivative=True)
            delta_w[l] = np.dot(delta_b[l], z_list[l].transpose())
        return delta_w, delta_b

    def train_batch(self, data_x, data_y):
        delta_w = [np.zeros_like(w) for w in self.weights]
        delta_b = [np.zeros_like(b) for b in self.biases]
        count = 0
        for x, y in zip(data_x, data_y):
            d_delta_w, d_delta_b = self.backward(x, y)
            delta_w = [np.add(d, dd) for d, dd in zip(delta_w, d_delta_w)]
            delta_b = [np.add(d, dd) for d, dd in zip(delta_b, d_delta_b)]
            count += 1
        return delta_w, delta_b

    def train(self, data_x, data_y, eta, l2):
        for epoch in range(EPOCH):
            print('------')
            print(epoch)
            print('EPOCH')
            dataset = list(zip(data_x, data_y))
            np.random.shuffle(dataset)
            x_shuffled, y_shuffled = zip(*dataset)
            x_shuffled = np.array(x_shuffled)
            y_shuffled = np.array(y_shuffled)
            x_validate, x_train = x_shuffled[:VALIDATION_SIZE], x_shuffled[VALIDATION_SIZE:]
            y_validate, y_train = y_shuffled[:VALIDATION_SIZE], y_shuffled[VALIDATION_SIZE:]
            x_train_batches = np.array(
                [x_train[s:s + BATCH_SIZE] for s in range(0, len(x_train) - BATCH_SIZE, BATCH_SIZE)])
            y_train_batches = np.array(
                [y_train[s:s + BATCH_SIZE] for s in range(0, len(y_train) - BATCH_SIZE, BATCH_SIZE)])
            for x_batch, y_batch in zip(x_train_batches, y_train_batches):
                delta_w, delta_b = self.train_batch(x_batch, y_batch)
                self.biases = [b - (eta / len(x_batch)) * nb for b, nb in zip(self.biases, delta_b)]
                self.weights = [(1 - eta * l2 / len(x_train)) * w - (eta / len(x_batch)) * nw for w, nw in
                                zip(self.weights, delta_w)]
                print('loss: ' + str(self.test_loss(x_validate, y_validate)))

    def test_loss(self, validate_x, validate_y):
        count = 0
        for x, y in zip(validate_x, validate_y):
            z_list, _ = self.forward(x)
            if np.argmax(z_list[-1]) == y:
                count += 1
        return count / len(validate_x)


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        s = 1 / (1 + np.exp(-x))
        return s


def softmax(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        e = np.exp(x - np.max(x))
        s = e / np.sum(e)
        return s


def cross_entropy_loss(out, label):
    return -np.sum([l * np.nan_to_num(np.log(y)) for y, l in zip(out, label)])


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def main():
    train_data = extract_data('dataset/train-images-idx3-ubyte.gz', 60000)
    train_labels = extract_labels('dataset/train-labels-idx1-ubyte.gz', 60000)
    test_data = extract_data('dataset/t10k-images-idx3-ubyte.gz', 10000)
    test_labels = extract_labels('dataset/t10k-labels-idx1-ubyte.gz', 10000)
    model = MLP(LAYERS)
    model.train(train_data, train_labels, LEARNING_RATE, L2_NORM)
    print(model.test_loss(test_data, test_labels))


if __name__ == '__main__':
    main()
