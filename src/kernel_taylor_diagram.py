"""
Taylor diagram statistics with classic, kernel, MMD and HSIC inner products.

Each estimator defines an inner product <X, Y>; the Taylor diagram values
(std, correlation, centered RMSE) are then derived from it in `compute_td`.

TaylorEstimator is the entry point: it creates the estimator of the type given,
'classic', 'kernel_classic', 'kernel_mmd' or 'kernel_hsic'.
"""

import sys

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer


def _check_array(A, name):
    if type(A) is not np.ndarray:
        raise TypeError(f"Expected np.ndarray for {name}, got {type(A).__name__}")


def _kernel_params(kernel, gamma, degree, coef0, kernel_params):
    """Keyword arguments for sklearn's pairwise_kernels."""
    if callable(kernel):
        return kernel_params or {}
    params = {}
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params["gamma"] = gamma
    if kernel in ['poly']:
        params["degree"] = degree
    if kernel in ['poly', 'sigmoid']:
        params["coef0"] = coef0
    return params


def compute_td(inner_product_func, X, Y, normalized: bool):
    '''
        Computes Taylor diagram values from an inner product function.

        X and Y -- as defined in inner_product_func
    '''
    cov_XX = inner_product_func(X, X)
    cov_YY = inner_product_func(Y, Y)
    cov_XY = inner_product_func(X, Y)
    # If all three are small, it is the 'comparing zero vs. zero' case.
    # The normalized diagram will then return the reference point.
    if np.abs(cov_XX) <= 1e-10 and np.abs(cov_YY) <= 1e-10 and np.abs(cov_XY) <= 1e-10:
        cov_XX = cov_YY = cov_XY = 1e-10

    std_X = np.sqrt(cov_XX)
    std_Y = np.sqrt(cov_YY)
    corr = cov_XY / (std_X * std_Y)
    mse = cov_XX + cov_YY - 2. * cov_XY
    # Numerical MSE can be slightly below 0 (e.g. for constant time series).
    if mse >= -1e-10:
        mse = np.maximum(mse, 0)
    if mse >= 0:
        rmse = np.sqrt(mse)
    else:
        print(f"Computed MSE ({mse}) is not semi-positive. Making RMSE=sqrt(abs(MSE)):",
              file=sys.stderr)
        rmse = np.sqrt(np.abs(mse))

    if normalized:
        std_Y /= std_X
        rmse /= std_X
        std_X = 1.
    # Numerical correlation can be slightly above 1.
    if np.abs(corr) <= 1.01:
        corr = np.clip(corr, -1., 1.)
    return {'std_X': std_X, 'std_Y': std_Y, 'corr': corr, 'rmse': rmse}


class ClassicEstimator:
    '''
        Computing classic Taylor diagram values for:

        X: [n_samples, ...]

        Y: [n_samples, ...]
    '''
    def __init__(self, centered=True, normalized=False, **kwargs):
        self.centered = centered
        self.normalized = normalized

    def inner_product(self, X: np.ndarray, Y: np.ndarray):
        _check_array(X, 'X')
        _check_array(Y, 'Y')
        if self.centered:
            X = X - X.mean(axis=0)
            Y = Y - Y.mean(axis=0)
        return (X * Y).mean()

    def get_feature_objects(self, X, Y):
        return X, Y

    def compute(self, X: np.ndarray, Y: np.ndarray):
        '''
            X and Y -- as defined in inner_product
        '''
        X, Y = self.get_feature_objects(X, Y)
        return compute_td(self.inner_product, X, Y, normalized=self.normalized)


class _KernelEstimatorBase(ClassicEstimator):
    '''
        Common part of the kernel estimators: the kernel and its parameters,
        passed to sklearn's pairwise_kernels. The subclasses define the inner product.
    '''
    def __init__(self, normalized=False, kernel='rbf', gamma=None, degree=3, coef0=1,
                 kernel_params=None):
        super().__init__(centered=None, normalized=normalized)
        self.kernel = kernel
        self.kernel_params = _kernel_params(kernel, gamma, degree, coef0, kernel_params)

    def _kernel(self, X, Y):
        return pairwise_kernels(X, Y, metric=self.kernel, **self.kernel_params)


class KernelClassicEstimator(_KernelEstimatorBase):
    '''
        Computing Taylor diagram based on the kernel inner product:
        mean of the diagonal of the centered cross-kernel matrix K(X, Y),
        i.e. covariance of paired samples in the RKHS.

        'centered' argument is ignored

        X: [n_samples, n_features]

        Y: [n_samples, n_features]
    '''
    def inner_product(self, X: np.ndarray, Y: np.ndarray):
        _check_array(X, 'X')
        _check_array(Y, 'Y')
        K = KernelCenterer().fit_transform(self._kernel(X, Y))
        return np.diag(K).mean()


class KernelMMDEstimator(_KernelEstimatorBase):
    '''
        Computing Taylor diagram based on the MMD inner product:
        inner product of the kernel mean embeddings of X and Y.
        RMSE is then the MMD between the two distributions.

        'centered' argument is ignored

        X: [n_samples, n_features]

        Y: [n_samples, n_features]
    '''
    def inner_product(self, X: np.ndarray, Y: np.ndarray):
        _check_array(X, 'X')
        _check_array(Y, 'Y')
        return self._kernel(X, Y).mean()


class KernelHSICEstimator(_KernelEstimatorBase):
    '''
        Computing Taylor diagram based on the HSIC inner product:
        <X, Y> = HSIC(X, Y) = tr(KX KY) / n^2 with centered kernel matrices KX and KY.
        The correlation is the centered kernel alignment (CKA). The values are in
        squared units of the kernel, as HSIC compares kernel matrices.

        The same kernel is used for X and Y. X and Y may still have different numbers
        of features, as each kernel matrix compares the samples of one dataset only.

        'centered' argument is ignored

        X: [n_samples, n_features_X]

        Y: [n_samples, n_features_Y]
    '''
    def get_feature_objects(self, X, Y):
        _check_array(X, 'X')
        _check_array(Y, 'Y')
        KX = KernelCenterer().fit_transform(self._kernel(X, X))
        KY = KernelCenterer().fit_transform(self._kernel(Y, Y))
        return KX, KY

    def inner_product(self, KX: np.ndarray, KY: np.ndarray):
        _check_array(KX, 'KX')
        _check_array(KY, 'KY')
        # tr(KX KY) = sum(KX * KY) for symmetric matrices: O(n^2) instead of O(n^3)
        return np.sum(KX * KY) / KX.shape[0]**2


class TaylorEstimator:
    '''
        Taylor diagram statistics of the type given:

            'classic'          ClassicEstimator
            'kernel_classic'   KernelClassicEstimator
            'kernel_mmd'       KernelMMDEstimator
            'kernel_hsic'      KernelHSICEstimator

        The other arguments are passed to the estimator of that type, so only the type is
        needed for the defaults, for example
            TaylorEstimator('kernel_hsic')
            TaylorEstimator('kernel_hsic', normalized=True, gamma=0.5)

        X: [n_samples, ...]

        Y: [n_samples, ...]
    '''
    TYPES = {'classic': ClassicEstimator,
             'kernel_classic': KernelClassicEstimator,
             'kernel_mmd': KernelMMDEstimator,
             'kernel_hsic': KernelHSICEstimator}

    def __init__(self, td_type, **kwargs):
        if td_type not in self.TYPES:
            raise ValueError(f"Unknown Taylor diagram type {td_type!r}; "
                             f"expected one of: {', '.join(self.TYPES)}")
        self.td_type = td_type
        self.estimator = self.TYPES[td_type](**kwargs)

    def get_feature_objects(self, X, Y):
        return self.estimator.get_feature_objects(X, Y)

    def inner_product(self, X, Y):
        '''
            X and Y -- as returned by get_feature_objects
        '''
        return self.estimator.inner_product(X, Y)

    def compute(self, X: np.ndarray, Y: np.ndarray):
        '''
            Taylor diagram values: 'std_X', 'std_Y', 'corr' and 'rmse'
        '''
        return self.estimator.compute(X, Y)


# Old names, kept as aliases for compatibility with existing code: each one is the same
# object as its new name, so both names can be used and isinstance works with either.
compute_TD = compute_td
TD_Estimator = ClassicEstimator
Kernel_TD_Estimator = KernelClassicEstimator
MMD_TD_Estimator = KernelMMDEstimator
HSIC_TD_Estimator = KernelHSICEstimator