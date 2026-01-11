import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_gradient(x):
    return (x > 0).astype(float)

class MLP:
    def __init__(self, input_dim, layers=(2, 1), lr = 1e-3, epochs = 500, batch_size = 32, momentum = 0.9, l2 = 0.0, seed = 42, verbose=True):
        np.random.seed(seed)

        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.momentum = momentum
        self.verbose = verbose
        self.sizes = [input_dim] + list(layers)
        self.L = len(layers)

        self.W = []
        self.b = []
        self.W_vel = []
        self.b_vel = []
        for l in range(self.L):
            n_in, n_out = self.sizes[l], self.sizes[l+1]

            # He for Relu, Xavier for Sigmoid
            mult = 2 if l < self.L - 1 else 1

            self.W.append(np.random.randn(n_in, n_out) * np.sqrt(mult / n_in))
            self.b.append(np.zeros(n_out,))
            self.W_vel.append(np.zeros_like(self.W[l]))
            self.b_vel.append(np.zeros_like(self.b[l]))

        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    def forward(self, X):
        a = [X.copy()]
        z = []
        for l in range(self.L):
            z.append(a[-1] @ self.W[l] + self.b[l])
            if l == self.L - 1:
                a.append(sigmoid(z[-1]))
            else:
                a.append(relu(z[-1]))
        return a, z

    def backprop(self, a, z, Y):
        m = Y.shape[0]
        grad_W = [None] * self.L
        grad_b = [None] * self.L

        delta = a[-1] - Y
        for l in reversed(range(self.L)):
            grad_W[l] = a[l].T @ delta / m + self.l2 * self.W[l]
            grad_b[l] = np.mean(delta, axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_gradient(z[l-1])
        return grad_W, grad_b

    def apply_gradient(self, grad_W, grad_b):
        for l in range(self.L):
            self.W_vel[l] = self.momentum * self.W_vel[l] - self.lr * grad_W[l]
            self.b_vel[l] = self.momentum * self.b_vel[l] - self.lr * grad_b[l]
            self.W[l] += self.W_vel[l]
            self.b[l] += self.b_vel[l]

    def loss(self, Y_pred, Y):
        eps = 1e-12
        p = np.clip(Y_pred, eps, 1 - eps)
        return -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    def fit(self, X, Y, X_val=None, Y_val=None, shuffle=True):
        m = Y.shape[0]
        for e in range(self.epochs):
            if shuffle:
                perm = np.random.permutation(m)
                X, Y = X[perm], Y[perm]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                Xi = X[start:end]
                Yi = Y[start:end]

                a, z = self.forward(Xi)
                grad_W, grad_b = self.backprop(a, z, Yi)
                self.apply_gradient(grad_W, grad_b)

            # diagnostics
            a_all, _ = self.forward(X)
            train_loss = self.loss(a_all[-1], Y)
            train_acc = np.mean((a_all[-1] > 0.5).astype(int) == Y)

            val_loss, val_acc = None, None
            if X_val is not None and Y_val is not None:
                av, _ = self.forward(X_val)
                val_loss = self.loss(av[-1], Y_val)
                val_acc = np.mean((av[-1] > 0.5).astype(int) == Y_val)

            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_acc)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            if self.verbose and ((e + 1) % 20 == 0 or e == 0 or (e + 1) == self.epochs):
                if val_loss is not None:
                    print(
                        f"Epoch {e + 1}/{self.epochs} — loss: {train_loss:.4f}, acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {e + 1}/{self.epochs} — loss: {train_loss:.4f}, acc: {train_acc:.4f}")

    def predict(self, X):
        a, _ = self.forward(X)
        return (a[-1] > 0.5).astype(int)

    def accuracy(self, X, Y):
        return np.mean(self.predict(X) == Y)