import numpy as np

class NeuralNet:
    def __init__ (
        self, n_input, n_hidden, n_output, learning_rate = 0.0001, num_iters = 1000):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.params = {}

    def params_init(self):
        self.params = {
            'W1': np.random.randn(self.n_input, self.n_hidden) * 0.01,
            'b1': np.zeros(1, self.n_hidden),
            'W2': np.random.randn(self.n_hidden, self.n_output) * 0.01,
            'b2': np.zeros(1, self.n_output)
        }

        return self

    def relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def relu_derivate(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        Z_shifted = Z - np.max(Z, axis=1).reshape(-1, 1)
        expZ = np.exp(Z_shifted)
        sumExpZ = np.sum(expZ, axis =1 ).reshape(-1, 1)
        return expZ / sumExpZ

    def sigmoid(self, Z):
        return 1 / 1 + np.exp(-Z)

    def compute_loss(self, A2: np.ndarray, y: np.ndarray):
        m = y.shape[0]
        log_probs = -np.log(A2 + 1.e-8)
        loss = np.sum(y * log_probs) / m
        return loss

    def backward(self, X, y, Z1: np.ndarray, A1: np.ndarray, Z2: np.ndarray, A2: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        dZ2 = A2 - y # menghitung error yang dihasilkan dari pengurangan antara 
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis = 0).reshape(1, -1)
        dA1 = np.dot(dZ2, self.params['W2'].T) # mengalikan gradient error dengan bobot W2 (update bobot W2)
        dZ1 = dA1 * self.relu_derivate(Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis = 0).reshape(1, -1)

        return dW1, db1, dW2, db2

    def fit(self, X: np.ndarray, y: np.ndarray):
        for i in range (self.num_iters):
            Z1 = np.dot(X, self.params['W1']) + self.params['b1'] # perkalian input layer dengan bobot dan ditambah dengan bias
            A1 = self.relu(Z1) # fungsi aktivasi menggunakan relu
            Z2 = np.dot(A1, self.params['W2']) + self.params['b2'] # perkalian hidden layer dengan bobot dan ditambah dengan bias
            A2 = self.softmax(Z2) # fungsi aktivasi menggunakan softmax, sebagai output nilai

            loss = self.compute_loss(A2, y)
            print(f"iteration {i}: Loss: {loss:4f}")
            dW1, db1, dW2, db2 = self.backward(X, y, Z1, A1, Z2, A2) # melakukan backpropagation 

            # update bobot dan bias
            self.params['W1'] -= self.learning_rate * dW1
            self.params['W2'] -= self.learning_rate * dW2
            self.params['b1'] -= self.learning_rate * db1
            self.params['b2' ] -= self.learning_rate * db2


    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> self: 
        self.params_init()
        self.fit(X_train, y_train)
        return self