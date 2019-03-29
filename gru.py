import numpy as np
from numba import jit

EPOCH = 1000
LEARNING_RATE = 0.1
OUTPUT_ROUND = 50
SEQUENCE_LEN = 50


class GRU(object):
    def __init__(self, text):
        vocab_size = len(sorted(list(set(text))))
        self.text = text
        self.h_size = vocab_size  # hidden layer size
        self.vocab_size = vocab_size  # types of chars

        self.Wz = np.random.rand(self.h_size, vocab_size) * 0.1 - 0.05
        self.Uz = np.random.rand(self.h_size, self.h_size) * 0.1 - 0.05
        self.bz = np.zeros((self.h_size, 1))

        self.Wr = np.random.rand(self.h_size, vocab_size) * 0.1 - 0.05
        self.Ur = np.random.rand(self.h_size, self.h_size) * 0.1 - 0.05
        self.br = np.zeros((self.h_size, 1))

        self.Wh = np.random.rand(self.h_size, vocab_size) * 0.1 - 0.05
        self.Uh = np.random.rand(self.h_size, self.h_size) * 0.1 - 0.05
        self.bh = np.zeros((self.h_size, 1))

        self.Wy = np.random.rand(vocab_size, self.h_size) * 0.1 - 0.05
        self.by = np.zeros((vocab_size, 1))

    @jit
    def forward(self, data, target, hprev, output=False):
        # hprev is for recursively train the model.
        # Each substring is trained as a part of the whole string.
        x, z, r, h_hat, h, y, p = {}, {}, {}, {}, {-1: hprev}, {}, {}
        total_loss = 0
        ixes = []
        for char in range(len(data)):
            # Set up one-hot encoded input
            x[char] = np.zeros((self.vocab_size, 1))
            x[char][data[char]] = 1

            z[char] = sigmoid(np.dot(self.Wz, x[char]) + np.dot(self.Uz, h[char - 1]) + self.bz)
            r[char] = sigmoid(np.dot(self.Wr, x[char]) + np.dot(self.Ur, h[char - 1]) + self.br)

            h_hat[char] = tanh(np.dot(self.Wh, x[char]) + np.dot(self.Uh, np.multiply(r[char], h[char - 1])) + self.bh)
            h[char] = np.multiply(z[char], h[char - 1]) + np.multiply((1 - z[char]), h_hat[char])

            y[char] = np.dot(self.Wy, h[char]) + self.by

            p[char] = softmax(y[char])
            total_loss -= np.sum(np.log(p[char][target[char]]))
            if output:
                ix = np.random.choice(range(self.vocab_size), p=p[char].ravel())
                ixes.append(ix)

        return x, z, r, h_hat, h, y, p, total_loss, ixes

    @jit
    def backward(self, data, target, hprev, output=False):
        x, z, r, h_hat, h, y, p, loss, out = self.forward(data, target, hprev, output)
        dWy, dWh, dWr, dWz = np.zeros_like(self.Wy), np.zeros_like(self.Wh), np.zeros_like(self.Wr), np.zeros_like(
            self.Wz)
        dUh, dUr, dUz = np.zeros_like(self.Uh), np.zeros_like(self.Ur), np.zeros_like(self.Uz)
        dby, dbh, dbr, dbz = np.zeros_like(self.by), np.zeros_like(self.bh), np.zeros_like(self.br), np.zeros_like(
            self.bz)
        dhnext = np.zeros_like(h[0])

        for char in reversed(range(len(data))):
            dy = np.copy(p[char])
            dy[target[char]] -= 1

            dWy += np.dot(dy, h[char].transpose())
            dby += dy

            dh = np.dot(self.Wy.transpose(), dy) + dhnext
            dh_hat = np.multiply(dh, (1 - z[char]))
            dh_hat_l = dh_hat * tanh(h_hat[char], derivative=True)

            dWh += np.dot(dh_hat_l, x[char].transpose())
            dUh += np.dot(dh_hat_l, np.multiply(r[char], h[char - 1]).transpose())
            dbh += dh_hat_l

            drhp = np.dot(self.Uh.transpose(), dh_hat_l)
            dr = np.multiply(drhp, h[char - 1])
            dr_l = dr * sigmoid(r[char], derivative=True)

            dWr += np.dot(dr_l, x[char].transpose())
            dUr += np.dot(dr_l, h[char - 1].transpose())
            dbr += dr_l

            dz = np.multiply(dh, h[char - 1] - h_hat[char])
            dz_l = dz * sigmoid(z[char], derivative=True)

            dWz += np.dot(dz_l, x[char].transpose())
            dUz += np.dot(dz_l, h[char - 1].transpose())
            dbz += dz_l

            dh_fz_inner = np.dot(self.Uz.transpose(), dz_l)
            dh_fz = np.multiply(dh, z[char])
            dh_fhh = np.multiply(drhp, r[char])
            dh_fr = np.dot(self.Ur.transpose(), dr_l)

            dhnext = dh_fz_inner + dh_fz + dh_fhh + dh_fr

        return loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, h[len(data) - 1], out

    def train(self, eta):
        chars = sorted(list(set(self.text)))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}
        mdWy, mdWh, mdWr, mdWz = np.zeros_like(self.Wy), np.zeros_like(self.Wh), np.zeros_like(self.Wr), np.zeros_like(
            self.Wz)
        mdUh, mdUr, mdUz = np.zeros_like(self.Uh), np.zeros_like(self.Ur), np.zeros_like(self.Uz)
        mdby, mdbh, mdbr, mdbz = np.zeros_like(self.by), np.zeros_like(self.bh), np.zeros_like(self.br), np.zeros_like(
            self.bz)
        smooth_loss = -np.log(1.0 / self.vocab_size) * len(self.text)
        for epoch in range(EPOCH):
            count = 0
            hprev = np.zeros((self.vocab_size, 1))
            for seq_ix in range(0, len(self.text) - SEQUENCE_LEN - 1, SEQUENCE_LEN):
                count += 1
                data = [char_to_ix[ch] for ch in self.text[seq_ix:seq_ix + SEQUENCE_LEN]]
                target = [char_to_ix[ch] for ch in self.text[seq_ix + 1:seq_ix + 1 + SEQUENCE_LEN]]

                if count % OUTPUT_ROUND == 0:
                    loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, hprev, sample = \
                        self.backward(data, target, hprev, True)
                    smooth_loss = smooth_loss * 0.999 + loss * 0.001
                    print(f'Epoch {epoch}, round {count}, Loss: {loss}, Smooth loss: {smooth_loss}')
                    print(''.join(ix_to_char[ix] for ix in sample))
                    print('------')
                else:
                    loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, hprev, _ = \
                        self.backward(data, target, hprev)

                for param, dparam, mem in zip(
                        [self.Wy, self.Wh, self.Wr, self.Wz, self.Uh, self.Ur, self.Uz, self.by, self.bh, self.br,
                         self.bz],
                        [dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz],
                        [mdWy, mdWh, mdWr, mdWz, mdUh, mdUr, mdUz, mdby, mdbh, mdbr, mdbz]):
                    np.clip(dparam, -5, 5, out=dparam)
                    mem += dparam * dparam
                    param += -eta * dparam / np.sqrt(mem + 1e-8)


@jit
def sigmoid(x, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    else:
        return s


@jit
def softmax(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        e = np.exp(x - np.max(x))
        s = e / np.sum(e)
        return s


@jit
def tanh(x, derivative=False):
    if derivative:
        return 1 - x ** 2
    else:
        return np.tanh(x)


@jit
def cross_entropy_loss(out, label):
    return -np.sum([l * np.nan_to_num(np.log(y)) for y, l in zip(out, label)])


def main():
    data = open('dataset/JaneEyre.txt', 'r').read()
    model = GRU(data)
    model.train(LEARNING_RATE)


if __name__ == '__main__':
    main()
