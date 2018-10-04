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


cdef kl_divergence(np.ndarray[DTYPE_t, ndim=1] pdf1, np.ndarray[DTYPE_t, ndim=1] pdf2):
    cdef DTYPE_t KL_div = 0
    cdef unsigned int i
    for i in range(pdf1.shape[0]):
        KL_div += (pdf1[i] * log2_stable_scalar(pdf1[i] / (pdf2[i] + 1e-31)))

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



cdef js_divergence_scalar(np.ndarray[DTYPE_t, ndim=1]  pdf1, np.ndarray[DTYPE_t, ndim=1]  pdf2, DTYPE_t pi1, DTYPE_t pi2):

    cdef np.ndarray[DTYPE_t, ndim=1] p_tilde_mat = np.empty_like(pdf1, dtype = DTYPE)
    cdef int i
    cdef DTYPE_t JS_div


    for i in range(pdf1.shape[0]):
        p_tilde_mat[i] = pi1 * pdf1[i] + pi2 * pdf2[i]

    JS_div = pi1 * kl_divergence(pdf1, p_tilde_mat) + pi2 * kl_divergence(pdf2, p_tilde_mat)

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

def calc_merger_cost_sIB(np.ndarray[DTYPE_t, ndim=1]  p_t,np.ndarray[DTYPE_t, ndim=2] p_x_given_t):

    cdef DTYPE_t p_t_bar, pi1, pi2
    cdef int i, ind_min
    cdef card_T = p_x_given_t.shape[0]
    cdef DTYPE_t last_p_t = p_t[card_T-1]
    cdef DTYPE_t min_cost = 10000
    cdef DTYPE_t costs
    cdef np.ndarray[DTYPE_t, ndim=1] cost_vec = np.empty(card_T-1)
    cdef np.ndarray[DTYPE_t, ndim=1] p_x_given_t1 = p_x_given_t[card_T-1, :]

    for i in range(card_T-1):
        p_t_bar = last_p_t + p_t[i]

        pi1 = last_p_t / p_t_bar

        pi2 = p_t[i] / p_t_bar

        #cost_vec[i] = p_t_bar * js_divergence_scalar(p_x_given_t1, p_x_given_t[i, :], pi1, pi2)
        costs = p_t_bar * js_divergence_scalar(p_x_given_t1, p_x_given_t[i, :], pi1, pi2)
        if (costs < min_cost):
            ind_min = i
            min_cost = costs

    return ind_min, min_cost #cost_vec #

def find_last_elem (np.ndarray[DTYPE_t, ndim=2] p_t_given_y, int border_between_clusters):
    cdef int last_elem, i
    last_elem = 0
    for i in range(p_t_given_y.shape[0]):
        if p_t_given_y[i,border_between_clusters] == 1 :
            last_elem = i
    return last_elem

cdef not_done_left_to_right_function(np.ndarray[DTYPE_t, ndim=2] p_t_given_y,
                                    np.ndarray[DTYPE_t, ndim=1] p_t,
                                    np.ndarray[DTYPE_t, ndim=2] p_x_y,
                                    np.ndarray[DTYPE_t, ndim=1] p_y,
                                    int cardinality_Y,
                                    np.ndarray[DTYPE_t, ndim=1] partner_last_cluster_vec,
                                    int border_between_clusters ,
                                    np.ndarray[DTYPE_t, ndim=1] last_cluster_vec,
                                    int cur_card_T):
    cdef bool done_left_to_right = False
    cdef bool empty_cluster
    cdef np.ndarray[DTYPE_t, ndim=2] p_t_given_y_cur_card
    cdef np.ndarray[DTYPE_t, ndim=2] p_x_given_t = np.empty([p_t.shape[0] , p_x_y.shape[1] ], dtype= DTYPE)
    cdef int i, cur_sum, last_elem
    cdef int old_cluster
    cdef int cardinality_T = p_t.shape[0]
    cdef int cardinality_X = p_x_y.shape[1]

    while not done_left_to_right:
        done_left_to_right = True

        last_elem = 0
        for i in range(cardinality_Y):
            if p_t_given_y[i,border_between_clusters] == 1 :
                last_elem = i

        old_cluster = border_between_clusters

        empty_cluster = True
        cur_sum = 0
        for i in range(cardinality_Y):
            if p_t_given_y[i, old_cluster] > 0:
                cur_sum += 1
            if cur_sum > 1:
                empty_cluster = False
                break

        if not empty_cluster:
            p_t_given_y[last_elem, :] = last_cluster_vec
            p_t_given_y[-(last_elem+1), :] = partner_last_cluster_vec

            cur_card_T += 2
            p_t_given_y_cur_card = p_t_given_y[:, :cur_card_T]

            # calculate p(t)  new
            for t in range(cardinality_T):
                p_t[t] = 0
                for y in range(cardinality_Y):
                    p_t[t] += p_t_given_y_cur_card[y,t] * p_y[y]

            # calculate p(x | t) new
            for t in range(cardinality_T):
                for x in range(cardinality_X):
                    p_x_given_t[t,x] = 0
                    for y in range(cardinality_Y):
                        p_x_given_t[t,x] += 1/p_t[t] * p_t_given_y[y,t] * p_x_y[y,x]

            #p_x_given_t = 1/(p_t[:cur_card_T, np.newaxis]) * np.dot(p_t_given_y_cur_card.T, p_x_y)

            ind_min, merger_costs_vec = calc_merger_cost(p_t, p_x_given_t, border_between_clusters,
                                              cur_card_T)

            if ind_min == 0:
                p_t_given_y[last_elem, border_between_clusters] = 1
                p_t_given_y[cardinality_Y-last_elem-1, cur_card_T-2-border_between_clusters-1] = 1

            else:
                p_t_given_y[last_elem, border_between_clusters+1] = 1
                p_t_given_y[- (last_elem+1), cur_card_T - 2 - border_between_clusters-2] = 1
                done_left_to_right = False

            p_t_given_y[-(last_elem+1), cur_card_T-1] = 0
            p_t_given_y[last_elem, - 2] = 0

            cur_card_T -= 2

    return p_t_given_y, p_t ,  p_x_given_t

cdef not_done_right_to_left_function(np.ndarray[DTYPE_t, ndim=2] p_t_given_y,
                                    np.ndarray[DTYPE_t, ndim=1] p_t,
                                    np.ndarray[DTYPE_t, ndim=2] p_x_y,
                                    np.ndarray[DTYPE_t, ndim=1] p_y,
                                    int cardinality_Y,
                                    np.ndarray[DTYPE_t, ndim=1] partner_last_cluster_vec,
                                    int border_between_clusters ,
                                    np.ndarray[DTYPE_t, ndim=1] last_cluster_vec,
                                    int cur_card_T):

    cdef bool done_right_to_left = False
    cdef bool empty_cluster
    cdef np.ndarray[DTYPE_t, ndim=2] p_t_given_y_cur_card
    cdef np.ndarray[DTYPE_t, ndim=2] p_x_given_t = np.empty([p_t.shape[0] , p_x_y.shape[1] ], dtype= DTYPE)
    cdef int i, cur_sum, first_elem
    cdef int old_cluster
    cdef int cardinality_T = p_t.shape[0]
    cdef int cardinality_X = p_x_y.shape[1]

    while not done_right_to_left:
        done_right_to_left = True

        # find first element in the cluster
        first_elem = 0
        for i in range(cardinality_Y):
            if p_t_given_y[i,border_between_clusters + 1] == 1 :
                first_elem = i
                break

        old_cluster = border_between_clusters + 1

        empty_cluster = True
        cur_sum = 0
        for i in range(cardinality_Y):
            if p_t_given_y[i, old_cluster] > 0:
                cur_sum += 1
            if cur_sum > 1:
                empty_cluster = False
                break

        if  not empty_cluster:
            p_t_given_y[first_elem, :] = last_cluster_vec
            p_t_given_y[cardinality_Y - first_elem - 1, :] = partner_last_cluster_vec

            cur_card_T += 2
            p_t_given_y_cur_card = p_t_given_y[:, :cur_card_T]

            # calculate p(t)  new
            for t in range(cardinality_T):
                p_t[t] = 0
                for y in range(cardinality_Y):
                    p_t[t] += p_t_given_y_cur_card[y,t] * p_y[y]



            # calculate p(x | t) new
            for t in range(cardinality_T):
                for x in range(cardinality_X):
                    p_x_given_t[t,x] = 0
                    for y in range(cardinality_Y):
                        p_x_given_t[t,x] += 1/p_t[t] * p_t_given_y[y,t] * p_x_y[y,x]



            ind_min, merger_costs_vec = calc_merger_cost(p_t, p_x_given_t,
                                                      border_between_clusters,
                                                      cur_card_T)

            if ind_min == 0:
                p_t_given_y[first_elem, border_between_clusters] = 1
                p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 2 - border_between_clusters - 1] = 1
                done_right_to_left = False
            else:
                p_t_given_y[first_elem, border_between_clusters + 1] = 1
                p_t_given_y[cardinality_Y - first_elem-1, cur_card_T - 2 - border_between_clusters-2] = 1

            p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 1] = 0
            p_t_given_y[first_elem, cur_card_T - 2] = 0
            cur_card_T -= 2
    return p_t_given_y, p_t ,  p_x_given_t


def lin_sim_algo_processing(np.ndarray[DTYPE_t, ndim=2] p_t_given_y,
                            np.ndarray[DTYPE_t, ndim=2] p_x_y,
                            np.ndarray[DTYPE_t, ndim=1] p_y,int cardinality_T, int cardinality_Y):

    cdef np.ndarray[DTYPE_t, ndim=2] init_mat = p_t_given_y
    cdef np.ndarray[DTYPE_t, ndim=2] end_mat = np.empty([cardinality_Y, cardinality_T + 2])
    cdef np.ndarray[DTYPE_t, ndim=1] p_t = np.empty(cardinality_T+2, dtype= DTYPE)
    cdef int cur_card_T = cardinality_T
    cdef int t, y, border_between_clusters


    # repeat until stable solution found
    last_cluster_vec = np.zeros(cardinality_T+2)
    last_cluster_vec[-2] = 1
    last_cluster_vec[-1] = 0
    partner_last_cluster_vec = np.zeros(cardinality_T+2)
    partner_last_cluster_vec[-2] = 0
    partner_last_cluster_vec[-1] = 1

    while not np.array_equal(init_mat, end_mat):
        for t in range(cardinality_T+2):
            p_t[t] = 0
            for y in range(cardinality_Y):
                p_t += p_t_given_y [y,t] * p_y[y]

        init_mat = p_t_given_y[:]

        for border_between_clusters in range(int(cardinality_T/2) - 1):
            done_left_to_right = False
            done_right_to_left = False

            p_t_given_y, p_t,p_x_given_t = not_done_left_to_right_function(p_t_given_y, p_t, p_x_y, p_y,cardinality_Y,
                                                      partner_last_cluster_vec, border_between_clusters ,
                                                      last_cluster_vec, cur_card_T)



            # check other direction
            p_t_given_y, p_t, p_x_given_t = not_done_right_to_left_function(p_t_given_y, p_t, p_x_y, p_y,
                                                  cardinality_Y,
                                                  partner_last_cluster_vec, border_between_clusters,
                                                  last_cluster_vec, cur_card_T)



        end_mat = p_t_given_y


    return p_t_given_y , p_t ,  p_x_given_t