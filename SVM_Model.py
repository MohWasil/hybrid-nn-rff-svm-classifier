import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, lr=0.001, lambda_pram=0.01, n_iter=1000, verbose=False, kernel='linear', gamma=0.1, degree=3, coef0=1,
                 C=1):
        self.weight = None
        self.bias = 0
        self.lr = lr
        self.lambda_pram = lambda_pram
        self.n_iter = n_iter
        self.verbose = verbose
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.multi_class = []
        self.models = []
        self.alpha = []
        self.support_vectors = []
        self.support_vector_label = []

    def fit(self, X, y):
        if self.gamma is None:
            self.gamma = 1 / X.shape[1]

        y = np.where(y <= 0, -1, 1)  # make sure labels are -1 and 1
        if self.kernel != 'linear':
            self._fit_dual(X, y)
            return

        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for ind, x_i in enumerate(X):
                y_i = y[ind]
                condition = y_i * (np.dot(self.weight, x_i) + self.bias)
                self._sgd_step(x_i, y_i, condition)

    def _fit_dual(self, X, y):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])

        # Convert to cvxopt format
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))

        G_std = np.diag(-np.ones(n_samples))  # -alpha <= 0
        h_std = np.zeros(n_samples)

        G_slack = np.diag(np.ones(n_samples))  # alpha <= C
        h_slack = np.ones(n_samples) * self.C

        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.hstack((h_std, h_slack)))

        A = matrix(y.astype(float), (1, n_samples))
        b = matrix(0.0)

        solvers.options['show_progress'] = self.verbose
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        support_vector_indices = alpha > 1e-5
        self.alpha = alpha[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_label = y[support_vector_indices]

        self.bias = 0
        for i in range(len(self.alpha)):
            self.bias += self.support_vector_label[i]
            self.bias -= np.sum(self.alpha * self.support_vector_label *
                                np.array([self._kernel_function(self.support_vectors[i], sv) for sv in
                                          self.support_vectors]))
        self.bias /= len(self.alpha)

    def predict(self, X):
        if self.kernel == 'linear' and self.weight is not None:
            return np.sign(np.dot(X, self.weight) + self.bias)

        y_pred = []
        for x in X:
            decision = 0
            for alpha_i, y_i, x_i in zip(self.alpha, self.support_vector_label, self.support_vectors):
                decision += alpha_i * y_i * self._kernel_function(x_i, x)
            decision += self.bias
            y_pred.append(np.sign(decision))

        return np.array(y_pred)

    def project(self, X):
        if self.kernel == 'linear' and self.weight is not None:
            return np.dot(X, self.weight) + self.bias

        result = []
        for x in X:
            decision = 0
            for alpha_i, y_i, x_i in zip(self.alpha, self.support_vector_label, self.support_vectors):
                decision += alpha_i * y_i * self._kernel_function(x_i, x)
            decision += self.bias
            result.append(decision)
        return np.array(result)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def _sgd_step(self, X_i, y_i, condition):
        if condition >= 1:
            self.weight -= self.lr * self.lambda_pram * self.weight
        else:
            self.weight -= self.lr * (self.lambda_pram * self.weight - y_i * X_i)
            self.bias += self.lr * y_i


    def _one_vs_rest(self, X, y):
        self.multi_class = np.unique(y)
        self.models = []

        for cls in self.multi_class:
            y_binary = np.where(y == cls, 1, -1)

            model = SVM(lr=self.lr, lambda_pram=self.lambda_pram, n_iter=self.n_iter, verbose=self.verbose,
                        kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0, C=self.C)

            model.fit(X, y_binary)

            self.models.append(model)

    def _one_vs_rest_predict(self, X):
        predictions = []

        for model in self.models:
            pred = model.project(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # shape: (n_classes, n_samples)

        best_class_indices = np.argmax(predictions, axis=0)

        return self.multi_class[best_class_indices]

    def _kernel_function(self, x_i, x_j):
        if self.kernel == 'linear':
            return np.dot(x_i, x_j)

        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x_i, x_j) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x_i - x_j) ** 2)

        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x_i, x_j) + self.coef0)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
