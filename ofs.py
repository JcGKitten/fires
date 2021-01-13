import numpy as np

class OFS:
    
    def __init__(self, regularization_param, step_size,
                 n_selected_ftr, n_total_ftr):
        """OFS algorithm for online features selection for binray data as
        proposed in "Online Feature Selection and Its Applications" by
        Wang et al 2014

        :param regularization_param: [description]
        :type regularization_param: float
        :param step_size: Step size for adjusting perceptron vectors
        :type step_size: float
        :param n_selected_ftr: Amount of features that should be selected
        :type n_selected_ftr: int
        :param n_total_ftr: Amount of features in the dataset
        :type n_total_ftr: int
        """        
        self.regularization_param = regularization_param
        self.step_size = step_size
        self.n_selected_ftr = n_selected_ftr
        self.w = np.zeros(n_total_ftr)
    

    def train(self, x, y):
        """Training function for the ofs algorithm, gets one trainig sample at 
        a time.

        :param x: observation with all features
        :type x: np.ndarray
        :param y: label, should be -1 or 1, 0 will be seen as -1
        :type y: int
        """        
        if y == 0:
            y = -1

        if np.dot(x, self.w) * y <= 1: # should be 0, shouldn't it
            w_tilde = (1-self.regularization_param * self.step_size)*self.w + \
                        self.step_size * y * x
            w_hat = min(1, (1/np.sqrt(self.regularization_param)) / \
                            np.linalg.norm(w_tilde) )*w_tilde
            self.w = self.__truncate(w_hat, self.n_selected_ftr)
        else:
            self.w *= (1-self.regularization_param*self.step_size)

    def __truncate(self, weights_array, B):
        w = np.zeros(len(weights_array))
        indices = np.argsort(weights_array)[::-1][:B-1]
        w[indices] = weights_array[indices]
        return w

    def get_feature_indices(self):
        return np.argsort(self.w)[::-1][:self.n_selected_ftr - 1]


class MC_OFS:
    def __init__(self, regularization_param, step_size, n_selected_ftr,
                 n_total_ftr, n_classes):
        """Extension of the known OFS algorithm by Wang et al for multiclass
        classification. Idea for it is a work in progress and motivated by
        multiclass perceptrons.

        :param regularization_param: [description]
        :type regularization_param: float
        :param step_size: Step size for learning new input
        :type step_size: flaot
        :param n_selected_ftr: amount of features that shall be selected
        :type n_selected_ftr: int
        :param n_total_ftr: total features of the dataset
        :type n_total_ftr: int
        :param n_classes: amount of classes wittin the dataset
        :type n_classes: n
        """        
        self.regularization_param = regularization_param
        self.step_size = step_size
        self.n_selected_ftr = n_selected_ftr
        self.w = np.zeros((n_classes, n_total_ftr))

    def train(self, x, y):
        """Training of the MC_OFS for one instance, class labels have to be from
        0 to k

        :param x: observation with all features
        :type x: np.ndarray
        :param y: class label
        :type y: int
        """        
        predictions = np.dot(self.w, x)
        print("Predictions: {}".format(predictions))
        prediction = np.where(predictions == np.amax(predictions))[0][0]
        print("Prediction: {}, class: {}".format(prediction, y))
        if y != prediction:
            print("{} \n {}".format(self.w[prediction], self.w[y]))
            #reduce wrong
            w_tilde = (1-self.regularization_param * self.step_size) * \
                       self.w[prediction] - self.step_size  * x
            w_hat = min(1, (1/np.sqrt(self.regularization_param)) / 
                        np.linalg.norm(w_tilde) ) * w_tilde
            self.w[prediction] = self.__truncate(w_hat, self.n_selected_ftr)

            #increase right
            w_tilde = (1-self.regularization_param * self.step_size) * \
                       self.w[y] + self.step_size * x
            w_hat = min(1, (1/np.sqrt(self.regularization_param)) / 
                        np.linalg.norm(w_tilde) )*w_tilde
            self.w[y] = self.__truncate(w_hat, self.n_selected_ftr)
        else:
            self.w[y] *= (1-self.regularization_param*self.step_size)

    def __truncate(self, weights_array, B):
        w = np.zeros(len(weights_array))
        indices = np.argsort(weights_array)[::-1][:B-1]
        w[indices] = weights_array[indices]
        return w

    def get_feature_indices(self):
        """Returns the indices of the selceted features, based on all features
        given.

        :return: Array with feature indices
        :rtype: np.ndarray
        """        
        w_mean = np.mean(self.w, axis=0)
        return np.argsort(w_mean)[::-1][:self.n_selected_ftr - 1]