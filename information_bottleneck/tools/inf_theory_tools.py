import numpy as np
__author__ = "Maximilian Stark"
__copyright__ = "18.05.2016, Institute of Communications,  Hamburg University of Technology"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Information Theory toolbox"
__doc__ = "This module contains a variety of small functions, useful to determine information theoretical quantities."


def log2_stable(value):
    """ Return the base 2 logarithm. Numerically stable is ensured by catching special cases, namely 0. """
    if np.any(value <= 0):
        if np.isscalar(value):
            return -1e6
        result = np.empty_like(value)
        result[value > 0] = np.log2(value[value > 0])
        result[value <= 0] = -1e6
        return result
    return np.log2(value)


def mutual_information(input_pdf):
    """ Return the Mutual Information for a given base an joint distribution sum over joint matrix over the rows
    to determine p_x.
    Args:
    input_pdf = a 2D array containing the joint probabilities. For p(x,y) x is fixed for a particular column and y is
                fixed in one row.
    """
    p_x = input_pdf.sum(0, keepdims=True)
    # sum over joint matrix over the columns to determine p_y
    p_y = input_pdf.sum(1, keepdims=True)
    MI = (input_pdf * log2_stable(input_pdf / (p_y.dot(p_x) + 1e-31))).sum()
    return MI


def kl_divergence(pdf1, pdf2):
    """ Return the Kullback-Leibler Divergence for two input PDFs. For use in IB algorithm.
    Note:
        The KL-divergence is not a real metric, the order of the input matters.
    Args:
    pdf1 =  a 2D array containing the joint probabilities. For p(x|y) x is fixed for a particular column and y is
            fixed in one row.
    pdf2 =  a 2D array containing the joint probabilities. For p(x|t) t is fixed for a particular column and x is
            fixed in one row.
    Note:
        It is also possible to input only an 1D array, which will be extended automatically using broadcasting.
        If the first pdf1 is a 1D arrays it is extended with pdf1[np.newaxis,:] to turn it in a row vector.
    """
    if pdf1.ndim == 1:
        pdf1 = pdf1[np.newaxis, :]
    KL_vec = (pdf1 * log2_stable(pdf1 / (pdf2 + 1e-31))).sum(1)
    return KL_vec




def js_divergence(pdf1, pdf2, pi1, pi2):
    """Return the Jenson-Shannen Divergence for two input PDFs and given pi1 and pi2.
    Note:
        The JS is the symmetrized and smoothed version of the KL-divergence.
    Args:
    pdf1 = a 2D array containing joint probabilities.
    pdf2 = a 2D array containing joint probabilities.
    pi1 = weighting factor
    pi2 = weighting factor
    """
    # catch special case that pi1 is a vector appears for sIB algorihtms
    if False in (np.isscalar(pi1), np.isscalar(pi2)):
        p_tilde_mat = pi1[:, np.newaxis] * pdf1 + pi2[:, np.newaxis] * pdf2
    else:
        p_tilde_mat = pi1 * pdf1 + pi2 * pdf2
    JS_vec = pi1 * kl_divergence(pdf1, p_tilde_mat) + pi2 * kl_divergence(pdf2, p_tilde_mat)
    return JS_vec

