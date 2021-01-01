import numpy as np
from warnings import warn
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


class FIRES:
    def __init__(self, n_total_ftr, target_values, mu_init=0, sigma_init=1, penalty_s=0.01, penalty_r=0.01, epochs=1,
                 lr_mu=0.01, lr_sigma=0.01, scale_weights=True, model='probit', number_monte_carlo_samples=10000,
                 class_probabilities=None):
        """
        FIRES: Fast, Interpretable and Robust Evaluation and Selection of features

        cite:
        Haug et al. 2020. Leveraging Model Inherent Variable Importance for Stable Online Feature Selection.
        In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20),
        August 23–27, 2020, Virtual Event, CA, USA.

        :param n_total_ftr: (int) Total no. of features
        :param target_values: (np.ndarray) Unique target values (class labels)
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
            #TODO: handle case with only one given obs better
            if y.shape == ():
                self.__softmax(x, y)
            else:
                for idx, label in enumerate(y):
                    self.__softmax(x, y)

        elif self.model == "regression":
            #TODO: condition rewrite
            if y.shape == ():
                self.__regression(x, y)
        # ### ADD YOUR OWN MODEL HERE ##################################################
        # elif self.model == 'your_model':
        #    self.__yourModel(x, y)
        ################################################################################
        else:
            raise NotImplementedError('The given model name does not exist')

        # Limit sigma to range [0, inf]
        if sum(n < 0 for n in self.sigma) > 0:
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

    def __softmax(self, x, y):  # needs self in model
        """
        Update the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
        Here we assume a multinominal distributed target variable. We use a Multinominal model as our base model.


        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type integer e.g. 1,2,3,4 etc.
        """
        for epoch in range(self.epochs):  # changed to self.epoch in model

            try:
                # l number of samples, j features, c classes
                # create 3d array with all r for current observation for multiple observation calculation we would need 4d array
                # r^cl_j = r[l, j, c] lxjxc
                r = np.random.randn(
                    self.n_mc_samples, self.n_total_ftr, self.model_param["no_classes"])
                # we only change the psi for the actuall given class still need all classes of course
                # r = np.random.randn(monte_carlo, n_total_ftr)
                print(r.shape)
                # calculate thetas for all samples and classes theta^cl_jt = theta[l,j,c]
                # lxjxc
                theta = r * self.sigma + self.mu
                print(theta.shape)
                # calculate all the etas
                # multiply all ftr_cols with given ftr_vector x
                eta = np.einsum("ljc,j->ljc", theta, x)
                # sum up all theta^cl_j * x_tj so we got l samples for all c classes
                eta = np.einsum("ljc->lc", eta)
                eta = np.exp(eta)  # we only need them exp
                # sum up etas for the l samples
                eta_sum = np.einsum("lc->l", eta)
                print(eta.shape)
                print(eta_sum.shape)
                # calculate softmax only for observed class
                # observation_etas = eta[:,y] with more observations we need transposition
                obs_eta = eta[:, y]
                softmax_lh = obs_eta / eta_sum  # lxo o is amount of given observations
                print(softmax_lh.shape)
                # marginal = np.einsum("lo->o", softmax_lh) / monte_carlo # 1xy
                marginal = np.sum(softmax_lh) / self.n_mc_samples
                print(marginal.shape)
                print(marginal)
                # calculate derivatives nabla_mu, nabla_sigma must be handled better
                # first calculate softmax dtheta
                # x_eta means observations x times the beloning etas
                # x_eta = np.einsum("oj,ol->ojl", x, observation_etas)
                # softmax_derivative = np.einsum("ojl,ol->ojl", x_eta, (eta_sum-observation_etas))/(eta_sum**2)
                x_eta = np.einsum("j,l->jl", x, obs_eta)
                softmax_derivative = np.einsum(
                    "jl,l->jl", x_eta, (eta_sum - obs_eta)) / (eta_sum**2)
                print(softmax_derivative.shape)
                nabla_mu = np.einsum(
                    "jl->j", softmax_derivative) / self.n_mc_samples
                print(nabla_mu.shape)
                r_jc = r[:, :, y].T
                print(r_jc.shape)
                nabla_sigma = np.einsum(
                    "jl->j", softmax_derivative * r_jc) / self.n_mc_samples
                print(nabla_sigma.shape)
                # Update parameters
                self.mu[:, y] += self.lr_mu * (nabla_mu / marginal)
                self.sigma[:, y] += self.lr_sigma * (nabla_sigma / marginal)

            except TypeError as e:
                raise TypeError(
                    'All features must be a numeric data type.') from e

    def __regression(self, x, y):
        """
         Update the distribution parameters mu and sigma by optimizing them in terms of the likelihood.
         Here we assume a normal distributed target variable. We use a identity model as our base model.

        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type float
        """

        for epoch in range(self.epochs):
            # Shuffle the observations
            # problem if only one is given, handle later try catch or so
            n_obs = len(y)
            random_idx = np.random.permutation(len(y))
            x = x[random_idx]
            y = y[random_idx]

            # Iterative update of mu and sigma
            try:
                # has shape o: observations, l: samples, j: features
                r = np.random.randn(n_obs, self.n_mc_samples, self.n_total_ftr)
                print("R shape: {}".format(r.shape))
                # calculate thetas for all samples and observations
                theta = np.einsum("olj,j->olj", r, self.sigma) + self.mu

                # calculate marginal likelihood shape o
                marginal = np.einsum("olj,oj->olj", theta, x)  # theta *x
                # sum over l and j / n_mc_samples
                marginal = np.einsum("olj->o", marginal) / self.n_mc_samples
                print("Marginal shape: {}".format(marginal.shape))

                # calculate derivatives
                nabla_mu = x  # shape oxj
                # 'ij->i' sum over all rows
                nabla_sigma = x * (np.einsum("olj->oj", r) /
                                   self.n_mc_samples)  # shape oxj

                # update mu and sigma
                self.mu += self.lr_mu * np.mean(nabla_mu.T / marginal, axis=1)
                self.sigma += self.lr_sigma * \
                    np.mean(nabla_sigma.T / marginal, axis=1)
            except:
                #TODO: handle errors
                print('failed')

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
        mu, sigma = self.mu, self.sigma
        if len(mu.shape) == 2:  # multinominal case
            if self.model_param["class_probs"]:
                # TODO declare class_probabilit
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
    