from typing import Tuple, Union, Callable
import numpy as np # type: ignore


def initialize_parameters(
    X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    """
    rows = np.random.choice(np.arange(X.shape[0]),k,replace=False)
    MU = X[rows,:]
    
    X_centered = X[:,np.newaxis,:] - MU[np.newaxis, : , : ] # (m x 1 x n) - (1 x k x n) = (m x k x n)
    SIGMA = np.einsum('mki,mkj->kij', X_centered, X_centered) / X.shape[0] # equivalent to X.T @ X / X.shape[0] but for each k

    PI = np.ones(k)/k
    return MU,SIGMA,PI

def compute_sigma(X: np.ndarray, MU: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    X_centered = X[:,np.newaxis,:] - MU[np.newaxis, : , : ] # (m x 1 x n) - (1 x k x n) = (m x k x n)
    SIGMA = np.einsum('mki,mkj->kij', X_centered, X_centered) / X.shape[0]
    return SIGMA

def prob(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate the probability of x (a single
    data point or an array of data points,
    you have to take care both cases) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] or numpy.ndarray[numpy.ndarray[float]]
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float or numpy.ndarray[float]
    """
    n = mu.shape[0]
    if x.ndim == 1:
        x = x[np.newaxis,:] # 1 x n
    x_centered = x - mu
    sigma_inv = np.linalg.inv(sigma)
    det = np.linalg.det(sigma)
    prob =  np.sqrt((2 * np.pi) ** n * det)**-1 * np.exp(-0.5 * np.sum(x_centered @ sigma_inv * x_centered, axis=1)) 
    
    return prob.item() if prob.size == 1 else prob # just to return a single float if the size is 1 instead of an array

def E_step(
    X: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray, PI: np.ndarray, k: int
) -> np.ndarray:
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    m = X.shape[0]
    responsibility = np.zeros((k,m))
    for i in range(k):
        responsibility[i] = PI[i] * prob(X, MU[i], SIGMA[i])
    responsibility = responsibility / np.sum(responsibility, axis=0) # normalize by summing each column  
    return responsibility


def M_step(
    X: np.ndarray, r: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    """
    m, n = X.shape
    new_MU = np.zeros((k, n))
    new_SIGMA = np.zeros((k, n, n))
    new_PI = np.zeros(k)
    for i in range(k):

        new_MU[i] = np.sum(X * r[i][:, np.newaxis], axis=0) / np.sum(r[i]) # weighted average of the data points: X(m,n) * r(k,m) choose each assignment (k), expand newaxis to apply to each dimension of X, normalize by the sum of the assignments


        X_centered = X - new_MU[i] 
        new_SIGMA[i] = (r[i][:, np.newaxis] * X_centered).T @ X_centered / np.sum(r[i]) # weighted average of the covariance matrix: X_centered(m,n) * r(k,m) choose each assignment (k), expand newaxis to apply to each dimension of X_centered, normalize by the sum of the assignments
        
        new_PI[i] = np.sum(r[i]) / m
    return new_MU, new_SIGMA, new_PI


def likelihood(
    X: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray, PI: np.ndarray, k: int
) -> float:
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), log(sum((k=1 to K),
                                      mixing_k * N(x_n | mean_k,stdev_k))))

    Make sure you are using natural log, instead of log base 2 or base 10.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    log_likelihood = float
    """
    m = X.shape[0]
    likelihoods = np.zeros((k, m))
    for i in range(k):
        likelihoods[i] = PI[i] * prob(X, MU[i], SIGMA[i]) # calculate the likelihood for each component of k for each data point
    log_likelihood = np.sum(np.log(np.sum(likelihoods, axis=0))) # sum over all k components of the likelihood of each components (how well is each component represented in the model, expected probability), compute log of that and sum over all data points
    return log_likelihood # overall likelihood of the model


def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (
        abs(prev_likelihood) * 0.9 < abs(new_likelihood) < abs(prev_likelihood) * 1.1
    )

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap # if the convergence counter is greater than the cap, return True to stop the training


def train_model(
    X: np.ndarray,
    k: int,
    convergence_function: Callable = default_convergence,
    initial_values: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    above.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values is None:
        MU, SIGMA, PI = initialize_parameters(X, k)
    else:
        MU, SIGMA, PI = initial_values

    prev_likelihood = None
    new_likelihood = None
    conv_ctr = 0

    while True:
        r = E_step(X, MU, SIGMA, PI, k)
        MU, SIGMA, PI = M_step(X, r, k)
        new_likelihood = likelihood(X, MU, SIGMA, PI, k)
        #Check convergence
        if prev_likelihood is not None: # skip the first iteration
            conv_ctr, terminate = convergence_function(prev_likelihood, new_likelihood, conv_ctr)
            if terminate:
                break
        prev_likelihood = new_likelihood

    return MU, SIGMA, PI, r
