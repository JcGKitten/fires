import numpy as np


class MCP():
    def __init__(self, n_features, n_classes, epochs=1, learning_rate=1):
        """
        MCP: A multiclass perceptron for classification

        :param n_features: amount of features per observation
        :type n_features: int
        :param n_classes: amount of different classes
        :type n_classes: int
        :param epochs: how often shall the given trainig data shall
                       be used for trainig, defaults to 1
        :type epochs: int, optional
        :param learning_rate: learning rate for vector update, defaults to 1
        :type learning_rate: int, optional
        """        
        self.f = n_features
        self.l = n_classes
        self.N = epochs
        self.lr = learning_rate
        self.weights = self.__initialize_weights()

    def __initialize_weights(self):
        weights = np.zeros((self.l, self.f))
        return weights

    def prediction(self, data_vector):
        print(data_vector)
        predictions = np.dot(self.weights, data_vector)
        return np.where(predictions == np.amax(predictions))[0][0]

    def train(self, train):
        for i in range(self.N):
            print(i)
            print(train)
            for example in train:
                label = example[1]
                doc = example[0]
                comp_label = self.prediction(doc)
                print(f"comp_label: {comp_label}")
                if comp_label != label:
                    self.weights[comp_label] -= self.lr * doc
                    self.weights[label] += self.lr * doc
