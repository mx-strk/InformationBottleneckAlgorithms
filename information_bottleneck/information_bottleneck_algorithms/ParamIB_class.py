from information_bottleneck.information_bottleneck_algorithms.generic_IB_class import GenericIB
from information_bottleneck.tools import inf_theory_tools as inf_tool
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.tools as cl_tools
import pyopencl.array as cl_array
import os
import time
from mako.template import Template
import progressbar
from scipy.stats import norm
from ib_plot_functions import IB_plotter

from scipy.sparse import csr_matrix
__author__ = "Maximilian Stark"
__copyright__ = "28.01.2019, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__="This module contains the Parametric Information Bottleneck Algorithm"


class ParamIB(GenericIB):
    """This class can be used to perform the Parametric Information Bottleneck algorithm. Further information can be 
    found in
    
     [SLB19] M. Stark, J. Lewandowsky, and G. Bauch, “A Parametric Information Bottleneck Algorithm for Gaussian Random 
     Variables and Gaussian Mixtures,” in 12th International ITG Conference on Systems, Communications and Coding 2019 
     (SCC’2019), Rostock, Germany, 2019.
      """

    def __init__(self, p_x_y_,
                 card_T_,
                 posterior_mean,
                 posterior_var,
                 x_vec ,
                 sigma_x,
                 symmetric_init = True ,
                 nror=50,KL_gmm=False):

        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], nror)
        self.name = 'KL means IB'

        self.cardinality_X=p_x_y_.shape[1]
        self.cardinality_Y=p_x_y_.shape[0]

        self.p_y = self.p_x_y.sum(1)
        self.p_t = np.zeros(self.cardinality_T)
        self.x_vec = x_vec
        self.sigma_x = sigma_x

        self.symmetric_init = symmetric_init
        self.KL_gmm = KL_gmm

        # compute the posterior mean and variance of observation
        self.posterior_mean = posterior_mean
        self.posterior_var = posterior_var

    def D_Kl_gaussian(self, mu0, mu1, var_0, var_1):
        value = 1 / 2 * (var_0 / var_1 + (mu0 - mu1) ** 2 / var_1 - 1 + np.log(var_1 / var_0))
        value /= np.log(2)
        return value

    def entropy_gaussian(self,variance):
        # computes the entropy of a gaussian RV and returns it in bits
        entropy = 1/2 * np.log(2*np.pi*np.e * variance)
        entropy /= np.log(2)
        return entropy

    def kl_divergence_mat_col_gaussian(self):
        KL_div = np.zeros((self.cardinality_Y,self.cardinality_T))
        for c in range(self.cardinality_T):
            KL_div[:, c] = self.D_Kl_gaussian(self.posterior_mean,self.cluster_mean[c], self.posterior_var, self.cluster_var[c])

        return KL_div

    def kl_divergence_mat_col_gaussian_mixture(self,p_t_given_y):
        KL_div = np.zeros((self.cardinality_Y,self.cardinality_T))

        for c in range(self.cardinality_T):
            indices = np.where(p_t_given_y == c)[0]
            p_y_range = self.p_y[p_t_given_y == c] / self.p_y[p_t_given_y == c].sum()
            post_range = self.posterior_mean[indices]
            post_var = self.posterior_var#[indices]
            den = 0
            for i, weight in enumerate(p_y_range):
                cur_mean = post_range[i]
                cur_var = post_var#[i]
                den += weight * np.exp(-self.D_Kl_gaussian(self.posterior_mean, cur_mean, self.posterior_var, cur_var) * np.log(2) )
            ### approximation
            DKL_estim = -np.log(den) / np.log(2)

            KL_div[:, c] = DKL_estim

        return KL_div

    def allow_move(self,argmin_vec,p_t_given_y):

        for i in range(self.cardinality_Y):
            desired_new_cluster = argmin_vec[i]
            old_cluster = p_t_given_y[i]
            if (self.local_length_vec[old_cluster] > 1):

                self.local_length_vec[desired_new_cluster] += 1
                self.local_length_vec[old_cluster] -= 1
                p_t_given_y[i] = desired_new_cluster

        return p_t_given_y

    def update_cluster_mean_var(self,p_t_given_y):
        self.cluster_mean = np.zeros(self.cardinality_T)
        self.cluster_var = np.zeros(self.cardinality_T)

        for cluster_idx in range(self.cardinality_T):
            # calculate p(t)
            indices = np.where(p_t_given_y == cluster_idx)[0]
            self.p_t[cluster_idx] = self.p_y[indices].sum(0)

            # calculate normalized weigths
            p_y_range = self.p_y[p_t_given_y == cluster_idx] / self.p_y[p_t_given_y == cluster_idx].sum()
            post_mean_range = self.posterior_mean[p_t_given_y == cluster_idx]
            self.cluster_mean[cluster_idx] = (post_mean_range * p_y_range).sum()
            self.cluster_var[cluster_idx] = np.sum(p_y_range * ((post_mean_range - self.cluster_mean[cluster_idx]) ** 2
                                                                + self.posterior_var))

    def estimate(self,p_t_given_y):
        if not self.KL_gmm:
            argmin_vec = np.argmin(self.kl_divergence_mat_col_gaussian(), axis=1)
        else:
            argmin_vec = np.argmin(self.kl_divergence_mat_col_gaussian_mixture(p_t_given_y), axis=1)

        p_t_given_y = self.allow_move(argmin_vec,p_t_given_y)

        return p_t_given_y

    def sort_clusters_by_cog_idcs(self):
        cogs = self.cluster_mean
        order_idcs = np.argsort(cogs.squeeze())

        self.p_x_given_t[:, :] = self.p_x_given_t[order_idcs, :]
        self.p_t[:] = self.p_t[order_idcs]
        self.p_t_given_y[:, :] = self.p_t_given_y[:, order_idcs]

    def run_IB_algo(self):
        self.Param_IB_algo()
        self.sort_clusters_by_cog_idcs()

    def Param_IB_algo(self):

        # Initialization
        # number of identity matrices fitting inside p_t_givem_y
        neye = int(np.floor(self.cardinality_Y / (self.cardinality_T + 1)))
        # remaining rows that will be filled up with ones in the first row
        remainder = int((self.cardinality_Y - neye * self.cardinality_T))

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        counter_vec = np.zeros(self.nror)
        p_t_given_y_mats = np.zeros((self.cardinality_Y, self.cardinality_T, self.nror))
        p_t_mats = np.zeros((1, self.cardinality_T, self.nror))
        p_x_given_t_mats = np.zeros((self.cardinality_T, self.cardinality_X, self.nror))

        self.I_XT_mat = []
        self.I_YT_mat = []
        self.p_t_mat = []
        self.LUT_mat = []
        bar = progressbar.ProgressBar(widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])

        for run in bar(range(0, self.nror)):
            self.I_XT_mat.append([])
            self.I_YT_mat.append([])
            self.p_t_mat.append([])
            self.LUT_mat.append([])

            # Beginn initialization
            if self.symmetric_init:
                self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T))
                # Initialization of p_t_given_y
                # Use dirichlet distribution to sample a valid density
                # the ones vector indicate that The expected value for each cluster is 1/(cardinaltiy_T)
                # Multiplying with constant changes variance. 1 means very high variance creating asymmetric clusters.
                # In the end +1 is needed to ensure that
                alpha = np.ones(int(self.cardinality_T / 2)) * 1
                border_vec = np.ones(alpha.shape[0]) * self.cardinality_Y / 2
                while border_vec[:-1].cumsum().max() >= self.cardinality_Y / 2:
                    border_vec = np.floor(np.random.dirichlet(alpha, 1).transpose() * (self.cardinality_Y / 2))
                    border_vec[border_vec == 0] = 1

                border_vec = np.hstack([border_vec[:-1].cumsum(), self.cardinality_Y / 2]).astype(int)

                a = 0
                for t in range(0, int(self.cardinality_T / 2)):
                    self.p_t_given_y[a:border_vec[t], t] = 1
                    self.p_t_given_y[self.cardinality_Y - border_vec[t]:self.cardinality_Y - a, self.cardinality_T - t - 1] = 1
                    a = border_vec[t]

            else:
                self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T + 1))
                self.p_t_given_y[:int(neye * self.cardinality_T), :self.cardinality_T] = np.tile(np.eye(self.cardinality_T),
                                                                                                 (neye, 1))
                self.p_t_given_y[self.cardinality_Y - remainder:, 0] = np.ones(remainder)

                self.p_t_given_y = self.p_t_given_y[np.random.permutation(self.cardinality_Y), :]

            p_t_given_y= np.argmax(self.p_t_given_y,axis=1)

            self.update_cluster_mean_var(p_t_given_y)

            # Processing
            # the vector holds the number of elements per cluster
            self.local_length_vec = np.zeros(self.cardinality_T)
            for t in range(self.cardinality_T):
                self.local_length_vec[t] = np.sum(p_t_given_y == t)

            # Processing
            counter = 0
            # repeat until stable solution found

            # Processing
            init_mat = p_t_given_y
            end_mat = np.zeros(self.cardinality_Y)

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                counter += 1
                init_mat = p_t_given_y.copy()
                # estimation step
                p_t_given_y = self.estimate(p_t_given_y.copy())
                #update step
                self.update_cluster_mean_var(p_t_given_y)

                if self.KL_gmm:
                    for t in range(self.cardinality_T):
                        indices = np.where(p_t_given_y == t)[0]
                        # calculate p(t)
                        self.p_t[t] = self.p_y[indices].sum(0)
                        # calculate p(x|t)
                        self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)

                if not self.KL_gmm:
                    new_MI = self.entropy_gaussian(self.sigma_x) - np.sum(self.p_t * self.entropy_gaussian(self.cluster_var))
                else:
                    p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]
                    p_xt = p_xt / p_xt.sum()
                    new_MI = inf_tool.mutual_information(p_xt)

                p_t_given_y_temp = np.zeros((self.cardinality_Y, self.cardinality_T ))
                p_t_given_y_temp[np.arange(p_t_given_y.shape[0]), p_t_given_y] = 1
                p_ty = p_t_given_y_temp * self.p_y[:, np.newaxis]
                MI_YT = inf_tool.mutual_information(p_ty)
                self.I_XT_mat[run].append(new_MI)
                self.I_YT_mat[run].append(MI_YT)
                self.p_t_mat[run].append(self.p_t.copy())
                self.LUT_mat[run].append(self.p_t_given_y.copy())
                end_mat = p_t_given_y

            if not self.KL_gmm:
                delta_x = self.x_vec[1] - self.x_vec[0]
                for t in range(self.cardinality_T):
                    self.p_x_given_t[t,:] = norm.pdf(self.x_vec, loc=self.cluster_mean[t], scale=np.sqrt(self.cluster_var[t])) * delta_x
            else:
                for t in range(self.cardinality_T):
                    indices = np.where(p_t_given_y == t)[0]
                    # calculate p(t)
                    self.p_t[t] = self.p_y[indices].sum(0)
                    # calculate p(x|t)
                    self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)

            self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T ))
            self.p_t_given_y[np.arange(p_t_given_y.shape[0]), p_t_given_y] = 1
            counter_vec[run]=counter
            p_t_given_y_mats[:, :, run] = self.p_t_given_y
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t

            if not self.KL_gmm:
                I_TX[run] = self.entropy_gaussian(self.sigma_x) - np.sum(self.p_t * self.entropy_gaussian(self.cluster_var))
            else:
                p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]
                p_xt = p_xt/p_xt.sum()
                I_TX[run] = inf_tool.mutual_information(p_xt)

            ib_fct[run] = I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = inf_tool.mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]
        p_ty = self.p_t_given_y * self.p_y[:, np.newaxis]
        self.MI_YT = inf_tool.mutual_information(p_ty)
        self.update_cluster_mean_var(np.argmax(self.p_t_given_y,axis=1))
        self.I_XT_mat = self.I_XT_mat[winner]
        self.I_YT_mat = self.I_YT_mat[winner]
        self.p_t_mat = self.p_t_mat[winner]
        self.LUT_mat = self.LUT_mat[winner]
