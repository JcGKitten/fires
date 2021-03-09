import numpy as np
from warnings import warn
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

try:
    import cupy as cp
    cupy_available = True
except ModuleNotFoundError:
    warn("Cupy couldn't be imported, so using FIRES with Multiclass can be slow.")
    cupy_available = False


class FIRES:
    def __init__(self, n_total_ftr, target_values, mu_init=0, sigma_init=1, penalty_s=0.01, penalty_r=0.01, epochs=1,
                 lr_mu=0.01, lr_sigma=0.01, scale_weights=True, model='probit', number_monte_carlo_samples=100,
                 class_probabilities=None):
        """
        FIRES: Fast, Interpretable and Robust Evaluation and Selection of features

        cite:
        Haug et al. 2020. Leveraging Model Inherent Variable Importance for Stable Online Feature Selection.
        In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20),
        August 23–27, 2020, Virtual Event, CA, USA.

        :param n_total_ftr: (int) Total no. of features
        :param target_values: (np.ndarray) Unique target values (class labels) or None for regression
        :param mu_init: (int/np.ndarray) Initial importance parameter
        :param sigma_init: (int/np.ndarray) Initial uncertainty parameter
        :param penalty_s: (float) Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
        :param penalty_r: (float) Penalty factor for the regularization (corresponds to gamma_r in the paper)
        :param epochs: (int) No. of epochs that we use each batch of observations to update the parameters
        :param lr_mu: (float) Learning rate for the gradient update of the importance
        :param lr_sigma: (float) Learning rate for the gradient update of the uncertainty
        :param scale_weights: (bool) If True, scale feature weights into the range [0,1]
        :param model: (str) Name of the base model to compute the likelihood (default is 'probit')
        :param number_monte_carlo_samples: (int) amount of monte_carlo samples used for estimate marginal_likelihood
        :param class_probability: (np.ndarray) Probabilities of the classes, if not set, all classes are assumend
                                               equaly likely
        """

        self.n_total_ftr = n_total_ftr
        self.target_values = target_values
        self.penalty_s = penalty_s
        self.penalty_r = penalty_r
        self.epochs = epochs
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.scale_weights = scale_weights
        self.model = model
        self.n_mc_samples = number_monte_carlo_samples
        
        # Additional model-specific parameters
        self.model_param = {}

        # Probit model
        if self.model == 'probit' and tuple(target_values) != (-1, 1):
            if len(np.unique(target_values)) == 2:
                self.mu = np.ones(n_total_ftr) * mu_init
                self.sigma = np.ones(n_total_ftr) * sigma_init
                # Indicates that we need to encode the target variable into {-1,1}
                self.model_param['probit'] = True
                warn('FIRES WARNING: The target variable will be encoded as: {} = -1, {} = 1'.format(
                    self.target_values[0], self.target_values[1]))
            else:
                raise ValueError('The target variable y must be binary.')

        # Multinominal logit (softmax) model
        if self.model == 'softmax':
            if cupy_available:

                self.model_param["n_classes"] = len(target_values)
                # maybe check for scale of mu init
                self.mu = cp.ones(
                    (n_total_ftr, self.model_param["n_classes"]))*mu_init
                self.sigma = cp.ones(
                    (n_total_ftr, self.model_param["n_classes"]))*sigma_init
                if class_probabilities != None:
                    if sum(class_probabilities) != 1:
                        raise ValueError("Class probs don't sum up to 1")
                    elif len(class_probabilities) != self.model_param["n_classes"]:
                        raise Exception(
                            "Length of target_values and class_probilities don't match")
                    else:
                        self.model_param["class_probs"] = True
                        self.class_probabilities = class_probabilities
            else:
                warn("Without Cupy the softmax function can be very slow")
                self.model_param["n_classes"] = len(target_values)
                # maybe check for scale of mu init
                self.mu = np.ones(
                    (n_total_ftr, self.model_param["n_classes"]))*mu_init
                self.sigma = np.ones(
                    (n_total_ftr, self.model_param["n_classes"]))*sigma_init
                if class_probabilities != None:
                    if sum(class_probabilities) != 1:
                        raise ValueError("Class probs don't sum up to 1")
                    elif len(class_probabilities) != self.model_param["n_classes"]:
                        raise Exception(
                            "Length of target_values and class_probilities don't match")
                    else:
                        self.model_param["class_probs"] = True
                        self.class_probabilities = class_probabilities
                
        if self.model == "regression":
            self.mu = np.ones(n_total_ftr) * mu_init
            self.sigma = np.ones(n_total_ftr) * sigma_init
        # ### ADD YOUR OWN MODEL PARAMETERS HERE #######################################
        # if self.model == 'your_model':
        #    self.model_param['your_model'] = {}
        ################################################################################

    def weigh_features(self, x, y):
        """
        Compute feature weights, given a batch of observations and corresponding labels

        :param x: (np.ndarray) Batch of observations
        :param y: (np.ndarray) Batch of labels
        :return: feature weights
        :rtype np.ndarray
        """

        # Update estimates of mu and sigma given the predictive model
        if self.model == 'probit':
            self.__probit(x, y)
        elif self.model == 'softmax':
            if cupy_available:
                self.__softmax_cp(x, y)
            else:
                self.__softmax_np(x, y)
        elif self.model == "regression":
            self.__regression(x, y)
        # ### ADD YOUR OWN MODEL HERE ##################################################
        # elif self.model == 'your_model':
        #    self.__yourModel(x, y)
        ################################################################################
        else:
            raise NotImplementedError('The given model name does not exist')

        # Limit sigma to range [0, inf]
        if np.sum(self.sigma < 0) > 0:
            self.sigma[self.sigma < 0] = 0
            warn(
                'Sigma has automatically been rescaled to [0, inf], because it contained negative values.')

        # Compute feature weights
        return self.__compute_weights()

    def __probit(self, x, y):
        """
        Update the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
        Here we assume a Bernoulli distributed target variable. We use a Probit model as our base model.
        This corresponds to the FIRES-GLM model in the paper.

        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type binary, i.e. {-1,1} (bool, int or str will be encoded accordingly)
        """

        for epoch in range(self.epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(y))
            x = x[random_idx]
            y = y[random_idx]

            # Encode target as {-1,1}
            if 'probit' in self.model_param:
                y[y == self.target_values[0]] = -1
                y[y == self.target_values[1]] = 1

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(x, self.mu)
                rho = np.sqrt(1 + np.dot(x**2, self.sigma**2))

                # Gradients
                nabla_mu = norm.pdf(y/rho * dot_mu_x) * (y/rho * x.T)
                nabla_sigma = norm.pdf(
                    y/rho * dot_mu_x) * (- y/(2 * rho**3) * 2 * (x**2 * self.sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y/rho * dot_mu_x)

                # Update parameters
                self.mu += self.lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self.sigma += self.lr_sigma * \
                    np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError(
                    'All features must be a numeric data type.') from e

    def __softmax_cp(self, x, y):
        """
        Update the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
        Here we assume a multinominal distributed target variable. We use a Multinominal model as our base model.
        Funciton with cupy functions.

        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type integer e.g. 0,1,2,3,4 etc.
        """
        
        if len(x.shape) != 2:
            x = x.reshape(1,len(x))
    
        observed_classes = np.unique(y)

        for obs_class in observed_classes:
            observations_index = np.where(y == obs_class)[0]
            x_obs = cp.array(x[observations_index])
            n_obs = len(x_obs)
            #print("obs_class: {}, n obs: {}".format(obs_class, n_obs))

            for epoch in range(self.epochs):
                    
                    # Iterative update of mu and sigma
                    try:
                        # o number of obs, l number of samples, j features,
                        # c classes
                        
                        # r shape: oxlxjxc
                        r = cp.random.randn(n_obs, self.n_mc_samples,
                                            self.n_total_ftr,
                                            self.model_param["n_classes"])

                        # theta shape: oxlxjxc
                        theta = (r * self.sigma + self.mu)

                        # eta shape: oxlxc
                        # multiply all ftr_cols with given ftr_vector x
                        eta = cp.einsum("oljc,oj->oljc", theta, x_obs) 

                        # sum up all theta^cl_j * x_tj so we got l samples
                        # for all c classes
                        eta = cp.einsum("oljc->olc", eta) 

                        # get a for numerical stability, shape oxl
                        a = cp.amax(eta, axis=2) * -1

                        eta = cp.einsum("olc->col", eta) + a
                        eta = cp.einsum("col->olc", eta)

                        eta = cp.exp(eta) # we only need them exp

                        # eta_sum shape: oxl
                        eta_sum = cp.einsum("olc->ol", eta)
                        
                        # calculate softmax only for all classes
                        # divide all etas by eta_sum
                        softmax_all = np.einsum("olc,ol->olc", eta, (1/eta_sum))
                        
                        # marginal shape: o
                        marginal = cp.einsum("ol->o",
                                             softmax_all[:,:,obs_class]) / \
                                   self.n_mc_samples


                        # calculate softmax derivative to theta
                        softmax_c = softmax_all[:,:,obs_class]

                        # first calculate derivative for all as k != c
                        softmax_derivative = -1 * cp.einsum("oj,ol,olc->oljc",
                                                            (x_obs),
                                                            softmax_c,
                                                            softmax_all)

                        # then for observed class c
                        softmax_derivative_c = cp.einsum("oj,ol,ol->olj",
                                                         x_obs,
                                                         softmax_c,
                                                         (1-softmax_c))

                        softmax_derivative[:,:,:,obs_class] = \
                            softmax_derivative_c
                        
                        

                        nabla_mu = cp.einsum("oljc->ojc", softmax_derivative) /\
                                   self.n_mc_samples

                        r_jc = r[:,:,:,obs_class]
                        #print(r_jc.shape)
                        nabla_sigma = cp.einsum("oljc,olj->ojc",
                                                softmax_derivative,r_jc) / \
                                      self.n_mc_samples

                        nabla_mu = cp.einsum("ojc->jco", nabla_mu)
                        self.mu += self.lr_mu * \
                                                cp.einsum("jco->jc",
                                                          (nabla_mu/ marginal))

                        nabla_sigma = cp.einsum("ojc->jco", nabla_sigma)
                        self.sigma += self.lr_sigma * \
                                                   cp.einsum("jco->jc",
                                                             (nabla_sigma / 
                                                             marginal))

                    except TypeError as e:
                            raise TypeError('All features must be a numeric data type.') from e

    def __softmax_np(self, x, y):
        """
        Update the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
        Here we assume a multinominal distributed target variable. We use a Multinominal model as our base model.
        This is the function which is used if cupy isn't installed, not recommended!

        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type integer e.g. 0,1,2,3,4 etc.
        """
        
        if len(x.shape) != 2:
            x = x.reshape(1,len(x))
    
        observed_classes = np.unique(y)

        for obs_class in observed_classes:
            observations_index = np.where(y == obs_class)[0]
            x_obs = np.array(x[observations_index])
            n_obs = len(x_obs)
            #print("obs_class: {}, n obs: {}".format(obs_class, n_obs))

            for epoch in range(self.epochs):
                    
                    # Iterative update of mu and sigma
                    try:
                        # o number of obs, l number of samples, j features,
                        # c classes
                        
                        # r shape: oxlxjxc
                        r = np.random.randn(n_obs, self.n_mc_samples,
                                            self.n_total_ftr,
                                            self.model_param["n_classes"])

                        # theta shape: oxlxjxc
                        theta = (r * self.sigma + self.mu)
                        
                        # get d for preventig exploding gradients
                        d = 10**(-np.floor(np.log10(np.max(theta))))

                        # eta shape: oxlxc
                        # multiply all ftr_cols with given ftr_vector x
                        eta = np.einsum("oljc,oj->oljc", theta, x_obs) 
                        
                        # get d for preventig exploding gradients
                        d = 10**(-np.floor(np.log10(np.max(eta))))
                        # sum up all theta^cl_j * x_tj so we got l samples
                        # for all c classes
                        eta = d * np.einsum("oljc->olc", eta) 
                        eta = np.exp(eta) # we only need them exp

                        # eta_sum shape: oxl
                        eta_sum = np.einsum("olc->ol", eta)
                        
                        # calculate softmax only for observed class
                        # obs_eta shape: oxl
                        obs_eta = eta[:,:,obs_class]
                        
                        # softmax_lh shape: oxl
                        softmax_lh = obs_eta / eta_sum # 
                        
                        # marginal shape: o
                        marginal = np.einsum("ol->o", softmax_lh) / \
                                   self.n_mc_samples

                        # calculate softmax derivative to theta
                        softmax_derivative = np.einsum("oj,ol->olj",
                                                      (d*x_obs), softmax_lh)
                        
                        softmax_derivative = np.einsum("olj,ol->olj",
                                                       softmax_derivative,
                                                       (1-softmax_lh))

                        nabla_mu = np.einsum("olj->oj", softmax_derivative) / \
                                   self.n_mc_samples

                        r_jc = r[:,:,:,obs_class]
                        #print(r_jc.shape)
                        nabla_sigma = np.einsum("olj->oj",
                                                softmax_derivative * r_jc) / \
                                      self.n_mc_samples

                        self.mu[:,obs_class] += self.lr_mu * \
                                                np.einsum("jo->j",
                                                          (nabla_mu.T / marginal))
                        self.sigma[:,obs_class] += self.lr_sigma * \
                                                   np.einsum("jo->j",
                                                             (nabla_sigma.T / marginal))

                    except TypeError as e:
                            raise TypeError('All features must be a numeric data type.') from e

    def __regression(self, x, y):
        """
         Update the distribution parameters mu and sigma by optimizing them in terms of the likelihood.
         Here we assume a normal distributed target variable. We use a identity model as our base model.

        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type float
        """

        for epoch in range(self.epochs):
            # Shuffle the observations
            try:
                n_obs = len(y)
                random_idx = np.random.permutation(len(y))
                x = x[random_idx]
                y = y[random_idx]
            except:
                n_obs = 1
                x.reshape(1, len(x))

            # Iterative update of mu and sigma
            try:
                # has shape o: observations, l: samples, j: features
                r = np.random.randn(n_obs, self.n_mc_samples, self.n_total_ftr)

                theta = np.einsum("olj,j->olj", r, self.sigma) + self.mu

                marginal = np.einsum("olj,oj->olj", theta, x)  # theta *x
                marginal = np.einsum("olj->o", marginal) / self.n_mc_samples

                # calculate derivatives
                nabla_mu = x
                
                nabla_sigma = x * (np.einsum("olj->oj", r) /
                                   self.n_mc_samples)  

                # update mu and sigma
                self.mu += self.lr_mu * np.mean(nabla_mu.T / marginal, axis=1)
                self.sigma += self.lr_sigma * \
                    np.mean(nabla_sigma.T / marginal, axis=1)
            except TypeError as e:
                raise TypeError('All features must be a numeric data type.') from e

    '''
    # ### ADD YOUR OWN MODEL HERE #################################################
    def __yourModel(self):
        """ 
        Your own model description.
        
        :param x: (np.ndarray) Batch of observations
        :param y: (np.ndarray) Batch of labels
        """
        
        gradientMu = yourFunction()  # Gradient of the (log) likelihood with respect to mu
        gradientSigma = yourFunction()  # Gradient of the (log) likelihood with respect to sigma
        self.mu += self.lr_mu * gradientMu
        self.sigma += self.lr_sigma * gradientSigma
    ################################################################################
    '''

    def __compute_weights(self):
        """
        Compute optimal weights according to the objective function proposed in the paper.
        We compute feature weights in a trade-off between feature importance and uncertainty.
        Thereby, we aim to maximize both the discriminative power and the stability/robustness of feature weights.

        :return: feature weights
        :rtype np.ndarray
        """

        if type(self.mu) == cp.core.core.ndarray:
            mu = cp.asnumpy(self.mu)
            sigma = cp.asnumpy(self.sigma)
        else:
            mu, sigma = self.mu, self.sigma

        if len(mu.shape) == 2:  # multinominal case
            if "class_probs" in self.model_param:
                mu = np.sum(mu * self.class_probabilities, axis=1)
                sigma = np.sum(sigma * self.class_probabilities, axis=1)
            else:
                mu = np.mean(mu, axis=1)
                sigma = np.mean(sigma, axis=1)

        # Compute optimal weights
        weights = (mu**2 - self.penalty_s * sigma**2) / (2 * self.penalty_r)

        if self.scale_weights:  # Scale weights to [0,1]
            weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).flatten()

        return weights
