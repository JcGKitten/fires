import numpy as np

class OFSSGD:
    def __init__(self, reduction_threshold, reduction_value, n_total_ftrs,
                 regularization_param, step_size):
        """ OFSGD algorithm as proposed in "An online approach for feature
        selection for classification in big data" by Nazar and Senthilkumar, 
        2017

        :param reduction_threshold: Threshold for reduction of the feature
            weights, can be one value for all or an array with an individual
            for each ftr
        :type reduction_threshold: np.ndarray
        :param reduction_value: value for reducing the feature weight when it's
            under the threshold, can be set for all or individual
        :type reduction_value: np.ndarray
        :param n_total_ftrs: amount of features within dataset
        :type n_total_ftrs: int
        :param regularization_param: [description]
        :type regularization_param: float
        :param step_size: step size for learning new inputs
        :type step_size: float
        :raises ValueError: Error is thrown, if the length of the given 
            reduction thresholds is not the number of total ftrs or 1
        """        
        try:
            if len(reduction_threshold) == n_total_ftrs:
                self.vartheta = reduction_threshold
            else:
                msg = "threshold vector and amount of features is not matching"
                raise ValueError(msg)
        except TypeError as e:
            self.vartheta = np.ones(n_total_ftrs) * reduction_threshold
            
        self.sigma = reduction_value
        self.W = np.zeros(n_total_ftrs)
        self.regularization_param = regularization_param
        self.step_size = step_size

    def __ola(self, x, y):
        
        if np.dot(x, self.W) * y <= 1:
            w_tilde = (1-self.regularization_param * self.step_size) * \
                self.W + self.step_size * y * x
            w_hat = min(1, (1/np.sqrt(self.regularization_param)) / \
                np.linalg.norm(w_tilde) ) * w_tilde
            self.W = w_hat
        else:
            self.W *= (1-self.regularization_param*self.step_size)

    def __SGr(self):

        for i in range(len(self.W)):
            if self.W[i] > 0 and self.W[i] < self.vartheta[i]:
                self.W[i] = max(0, self.W[i] - self.sigma)
            elif self.W[i] < 0 and self.W[i] > -self.vartheta[i]:
                self.W[i] = min(0, self.W[i] + self.sigma)


    def train(self, x, y):
        """Train feature weights, one instance at a time.

        :param x: observation with all features
        :type x: np.ndarray
        :param y: label, should be -1 or 1, 0 is set to -1
        :type y: int
        """
        if y == 0:
            y = -1
            
        self.__ola(x,y)
        self.__SGr()
            
    def get_weights(self):
        """Returns all features, which weights aren't reduced to zero

        :return: Array of feature indices
        :rtype: np.ndarray
        """        
        return np.where(self.W != 0)[0]    
        # i suppose that all featurs with weight unequal zero are kept


class MC_OFSSGD:
    def __init__(self, reduction_threshold, reduction_value, n_total_ftrs,
                 regularization_param, step_size, n_classes):
        """Extension for multiclass use of OFSSGD. Class labels must be from
        0 to k for k classes.

        :param reduction_threshold: Threshold for reduction of the feature
            weights, can be one value for all or an array with an individual
            for each ftr
        :type reduction_threshold: np.ndarray
        :param reduction_value: value for reducing the feature weight when it's
            under the threshold, can be set for all or individual
        :type reduction_value: np.ndarray
        :param n_total_ftrs: amount of features within dataset
        :type n_total_ftrs: int
        :param regularization_param: [description]
        :type regularization_param: float
        :param step_size: step size for learning new inputs
        :type step_size: float
        :param n_classes: amount of different classes within 
        :type n_classes: int
        :raises ValueError: Error is thrown, if the length of the given 
            reduction thresholds is not the number of total ftrs or 1
        """        
        try:
            if len(reduction_threshold) == n_total_ftrs:
                self.vartheta = reduction_threshold
            else:
                msg = "threshold vector and amount of features is not matching"
                raise ValueError(msg)
        except TypeError as e:
            self.vartheta = np.ones(n_total_ftrs) * reduction_threshold

        self.sigma = reduction_value
        self.W = np.zeros((n_classes, n_total_ftrs))
        self.regularization_param = regularization_param
        self.step_size = step_size

    def __ola(self, x, y):
        predictions = np.dot(self.W, x)
        prediction = np.where(predictions == np.amax(predictions))[0][0]
        
        if y != prediction:
            #print("{} \n {}".format(self.W[prediction], self.W[y]))
            #reduce wrong
            w_tilde = (1-self.regularization_param * self.step_size) * \
                self.W[prediction] - self.step_size  * x
            w_hat = min(1, (1/np.sqrt(self.regularization_param)) / \
                        np.linalg.norm(w_tilde) )*w_tilde
            self.W[prediction] = w_hat

            #increase right
            w_tilde = (1-self.regularization_param * self.step_size) * \
                       self.W[y] + self.step_size * x
            w_hat = min(1, (1/np.sqrt(self.regularization_param)) / \
                        np.linalg.norm(w_tilde) )*w_tilde
            self.W[y] = w_hat
            self.__SGr(y, prediction)
        else:
            self.W[y] *= (1-self.regularization_param*self.step_size)
            self.__SGr(y)

    def __SGr(self, y , prediction=None):

        for i in range(len(self.W[y])):
            if self.W[y,i] > 0 and self.W[y,i] < self.vartheta[i]:
                self.W[y,i] = max(0, self.W[y,i] - self.sigma)
            elif self.W[y,i] < 0 and self.W[y,i] > -self.vartheta[i]:
                self.W[y,i] = min(0, self.W[y,i] + self.sigma)
        if prediction != None:
            for i in range(len(self.W[y])):
                if self.W[prediction,i] > 0 and \
                    self.W[prediction,i] < self.vartheta[i]:
                    self.W[prediction,i] = max(0, self.W[prediction,i] - 
                                               self.sigma)
                elif self.W[prediction,i] < 0 and \
                    self.W[prediction,i] > -self.vartheta[i]:
                    self.W[prediction,i] = min(0, self.W[prediction,i] + 
                                               self.sigma)


    def train(self, x, y):
        """Train feature weights, one instance at a time.

        :param x: observation with all features
        :type x: np.ndarray
        :param y: label, should be from 0 to k
        :type y: int
        """     
        self.__ola(x,y)
        # calling it now from __ola to get the updated vectors
        # self.__SGr()
            
    def get_feature_indices(self):
        """Returns all features, which weights aren't reduced to zero

        :return: Array of feature indices
        :rtype: np.ndarray
        """
        W_mean = np.mean(self.W, axis=0)
        return np.where(W_mean != 0)[0]    
        # i suppose that all featurs with weight unequal zero are kept