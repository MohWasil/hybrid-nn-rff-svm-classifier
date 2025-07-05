import numpy as np

class RFFTransformer:
    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.W = None
        self.b = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        # Generate random weights W from N(0, 2γ)
        self.W = np.random.normal(loc=0, scale=np.sqrt(2 * self.gamma),
                                  size=(self.n_components, n_features))

        # Generate biases b uniformly from [0, 2π]
        self.b = np.random.uniform(0, 2 * np.pi, size=self.n_components)

        return self

    def transform(self, X):
        projection = np.dot(X, self.W.T) + self.b
        return np.sqrt(2 / self.n_components) * np.cos(projection)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
