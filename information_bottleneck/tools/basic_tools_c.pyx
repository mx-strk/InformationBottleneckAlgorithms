# cython: profile=True
import numpy as np
from libc.math cimport log2
from cpython cimport bool
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function

cdef  log2_stable(np.ndarray[DTYPE_t, ndim=1] value):
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim = 1] result  = np.empty_like(value, dtype=DTYPE)
    for i in range(value.shape[0]):
        if value[i] <= 0:
            result[i] = -1e6
        else:
            result[i] = log2(value[i])
    return result

@cython.profile(False)
cdef inline log2_stable_scalar(DTYPE_t value):
    if value <= 0:
        return -1e6
    else:
        return log2(value)


def kl_divergence(np.ndarray[DTYPE_t, ndim=1] pdf1, np.ndarray[DTYPE_t, ndim=1] pdf2):
    cdef DTYPE_t KL_div = 0
    cdef unsigned int i
    for i in range(pdf1.shape[0]):
        KL_div += (pdf1[i] * log2_stable_scalar(pdf1[i] / (pdf2[i] + 1e-31)))

    return KL_div

def kl_divergence_mat_col(np.ndarray[DTYPE_t, ndim=2] pdf1, np.ndarray[DTYPE_t, ndim=2] pdf2):
    cdef unsigned int i,c,j
    cdef np.ndarray[DTYPE_t, ndim=2] KL_div = np.zeros((pdf1.shape[0],pdf2.shape[0]), dtype = DTYPE)

    for c in range(pdf2.shape[0]):
        for j in range(pdf1.shape[0]):
            for i in range(pdf1.shape[1]):
                KL_div[j,c] += (pdf1[j,i] * log2_stable_scalar(pdf1[j,i] / (pdf2[c,i] + 1e-31)))


    return KL_div

def kl_divergence_mat_same(np.ndarray[DTYPE_t, ndim=2] pdf1, np.ndarray[DTYPE_t, ndim=2] pdf2):
    cdef unsigned int i,c,j
    cdef np.ndarray[DTYPE_t, ndim=1] KL_div = np.zeros((pdf1.shape[0]), dtype = DTYPE)

    for j in range(pdf1.shape[0]):
        for i in range(pdf1.shape[1]):
            KL_div[j] += (pdf1[j,i] * log2_stable_scalar(pdf1[j,i] / (pdf2[j,i] + 1e-31)))

    return KL_div


def mutual_information(np.ndarray[DTYPE_t, ndim=2] input_pdf):
    cdef np.ndarray[DTYPE_t, ndim=1]  p_x
    cdef np.ndarray[DTYPE_t, ndim=1]  p_y
    p_x = input_pdf.sum(0)
    p_y = input_pdf.sum(1)
    cdef DTYPE_t MI = 0
    cdef unsigned int i
    cdef unsigned int j
    for i in range(input_pdf.shape[0]):
        for j in range(input_pdf.shape[1]):
            MI += (input_pdf[i,j] * log2_stable_scalar(input_pdf[i,j] / (p_y[i] * p_x[j] + 1e-31)))
    return MI



def js_divergence_scalar(np.ndarray[DTYPE_t, ndim=1]  pdf1, np.ndarray[DTYPE_t, ndim=1]  pdf2, DTYPE_t pi1, DTYPE_t pi2):

    cdef np.ndarray[DTYPE_t, ndim=1] p_tilde_mat = np.empty_like(pdf1, dtype = DTYPE)
    cdef int i
    cdef DTYPE_t JS_div


    for i in range(pdf1.shape[0]):
        p_tilde_mat[i] = pi1 * pdf1[i] + pi2 * pdf2[i]

    JS_div = pi1 * kl_divergence(pdf1, p_tilde_mat) + pi2 * kl_divergence(pdf2, p_tilde_mat)

    return JS_div

def js_divergence_mat(np.ndarray[DTYPE_t, ndim=2]  pdf1, np.ndarray[DTYPE_t, ndim=2]  pdf2, DTYPE_t pi1, DTYPE_t pi2):

    cdef np.ndarray[DTYPE_t, ndim=2] p_tilde_mat
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] JS_div

    p_tilde_mat = pi1 * pdf1 + pi2 * pdf2

    JS_div = pi1 * kl_divergence_mat_same(pdf1, p_tilde_mat) + pi2 * kl_divergence_mat_same(pdf2, p_tilde_mat)

    return JS_div


cdef inline int find_ind_min(DTYPE_t a, DTYPE_t b): return 1 if a > b else 0

def calc_merger_cost(np.ndarray[DTYPE_t, ndim=1]  p_t,np.ndarray[DTYPE_t, ndim=2] p_x_given_t,
                     int border_between_clusters, int cur_card_T_):

    cdef int bbc = border_between_clusters
    cdef int cur_card_T = cur_card_T_-2
    cdef DTYPE_t p_t_bar, pi1, pi2
    cdef int i, ind_min
    cdef DTYPE_t cost_vec[2]
    cdef np.ndarray[DTYPE_t, ndim=1] p_x_given_t1 = p_x_given_t[cur_card_T, :]

    for i in range(2):
        p_t_bar = p_t[cur_card_T] + p_t[bbc+i]
        pi1 = p_t[cur_card_T] / p_t_bar
        pi2 = p_t[bbc+i] / p_t_bar

        cost_vec[i] = p_t_bar * js_divergence_scalar(p_x_given_t1, p_x_given_t[bbc+i, :], pi1, pi2)

    ind_min = find_ind_min (cost_vec[0], cost_vec[1])

    return ind_min, cost_vec

def find_last_elem (np.ndarray[DTYPE_t, ndim=2] p_t_given_y, int border_between_clusters):
    cdef int last_elem, i
    last_elem = 0
    for i in range(p_t_given_y.shape[0]):
        if p_t_given_y[i,border_between_clusters] == 1 :
            last_elem = i
    return last_elem

