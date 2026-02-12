from typing import Callable

import numpy as np


def unscented_transform(
    func: Callable,
    mean: np.array,
    cov: np.array,
    alpha: float = 0.5,
    beta: float = 2,
    kappa: float = None,
) -> tuple[np.array, np.array]:
    """
    Performs the Unscented Transform on a given mean and covariance.

    Args:
        func (model): The model function to transform.
        mean (np.array): The mean vector (L x 1).
        cov (np.array): The covariance matrix (L x L).
        alpha (float): Spread of the sigma points.  Defaults to 0.5.
        beta (float): Incorporates prior knowledge of the distribution.  Defaults to 2.
        kappa (float): Secondary scaling parameter.  Defaults to 2 - L if None is provided, where L is the dimensionality of the mean.

    Returns:
        tuple: Transformed mean and covariance.
    """

    # Pre-calculate kappa tuning parameter if not provided
    L = mean.shape[0]
    output_dim = func(mean).shape[0]
    if kappa is None:
        kappa = 2 - L

    # Ensure it is valid to perform the unscented transform
    if L < 2:
        raise ValueError("Mean vector must have dimensionality of at least 2.")

    # Ensure covariance is in correct shape
    if cov.ndim == 1:
        cov = np.diag(cov)  # Allow providing diagonal variance as 1D array
    if cov.shape != (L, L):
        raise ValueError(
            "Covariance matrix shape does not match mean vector dimensionality."
        )

    # Compute scaling parameter lambda:
    lambda_ = alpha**2 * (L + kappa) - L

    # Generate sigma points and weights:
    sigma_points = np.zeros((2 * L + 1, L))
    weights_mean = np.zeros(2 * L + 1)
    weights_cov = np.zeros(2 * L + 1)

    sqrt_cov = np.linalg.cholesky((L + lambda_) * cov)
    sigma_points[0] = mean
    weights_mean[0] = lambda_ / (L + lambda_)
    weights_cov[0] = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)
    for i in range(L):
        sigma_points[i + 1] = mean + sqrt_cov[:, i]
        sigma_points[L + i + 1] = mean - sqrt_cov[:, i]
        weights_mean[i + 1] = weights_mean[L + i + 1] = 1 / (2 * (L + lambda_))
        weights_cov[i + 1] = weights_cov[L + i + 1] = 1 / (2 * (L + lambda_))

    # Propagate sigma points through the model:
    propagated_sigma_points = np.zeros((2 * L + 1, output_dim))
    for i in range(2 * L + 1):
        propagated_sigma_points[i] = func(sigma_points[i])

    # Compute the propagated mean:
    propagated_mean = np.zeros(propagated_sigma_points[0].shape)
    for i in range(2 * L + 1):
        propagated_mean += weights_mean[i] * propagated_sigma_points[i]

    # Compute the propagated covariance:
    propagated_covariance = np.zeros((output_dim, output_dim))
    for i in range(2 * L + 1):
        diff = propagated_sigma_points[i] - propagated_mean
        propagated_covariance += weights_cov[i] * np.outer(diff, diff)

    return propagated_mean, propagated_covariance
