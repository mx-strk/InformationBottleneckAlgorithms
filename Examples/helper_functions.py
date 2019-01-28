import numpy as np


def gaussian_over_prob(x, mu, sigma2, delta):
    prob = norm.sf((x - mu + delta / 2) / np.sqrt(sigma2))
    return prob


def gaussian_under_prob( x, mu, sigma2, delta):
    prob = 1 - gaussian_over_prob(x - delta, mu, sigma2, delta)
    return prob
