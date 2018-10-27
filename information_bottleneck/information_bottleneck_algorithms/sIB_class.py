import numpy as np

from information_bottleneck.information_bottleneck_algorithms.generic_IB_class import GenericIB
from information_bottleneck.tools import inf_theory_tools as inf_tool

__author__ = "Maximilian Stark"
__copyright__ = "22.05.2016, Institute of Communications, Hamburg University of Technology"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__="This module contains the sequential Information Bottleneck"


class sIB(GenericIB):
    """This class can be used to perform the sequential Information Bottleneck algorithm.
      Args:
      input parameter
          p_x_y                     input joint pdf, where x is the number of columns and y the number of rows
      IB related parameters
          cardinality_T
          beta                      is set to Inf for the sIB
          eps                       not used in this algorithm set to []
          nror                      number of runs
      Return:
      mutual information
          MI_XT                     mutual information of output I(X;T)
          MI_XY                     mutual information of input I(X;Y)
      output PDF_s
          p_t_given_y
          p_x_given_t
          p_t
      Note: The values can be accessed using the appropriate return functions of the class
        self.return_results         return all values
        self.display_MI             return the MIs graphically as well as percentage of maintained mutual information
      """

    def __init__(self, p_x_y_, card_T_, nror_):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], nror_)
        self.name = 'sequential IB'

    def run_IB_algo(self):
        self.sIB_algo()

    def sIB_algo(self):
        """ This function tries to minimize the information bottleneck functional using a sequential algorithm.
        This algorithm only allows for deterministic cluster mapping, meaning beta is always set to infinity."""
        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X=p_x.shape[0]
        cardinality_Y=p_y.shape[0]

        cur_card_T = self.cardinality_T

        # Initialization
        # number of identity matrices fitting inside p_t_givem_y
        neye = int(np.floor(cardinality_Y / (self.cardinality_T + 1)))
        # remaining rows that will be filled up with ones in the first row
        remainder = int((cardinality_Y - neye * self.cardinality_T))

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_YT = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.cardinality_T + 1, self.nror))
        p_t_mats = np.zeros((1, self.cardinality_T, self.nror))
        p_x_given_t_mats = np.zeros((self.cardinality_T, cardinality_X, self.nror))

        reset_vec = np.array([1,0])

        # run for-loop for each number of run
        for run in range(0, self.nror):
            self.printProgress(run, self.nror, prefix='Run:')
            self.p_t_given_y = np.zeros((cardinality_Y, self.cardinality_T + 1))
            self.p_t_given_y[:int(neye * self.cardinality_T), :self.cardinality_T] = np.tile(np.eye(self.cardinality_T), (neye, 1))

            self.p_t_given_y[cardinality_Y - remainder:, 0] = np.ones(remainder)
            self.p_t_given_y = self.p_t_given_y[np.random.permutation(cardinality_Y), :]

            # Processing
            init_mat = self.p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.cardinality_T + 1))
            ind_min = 0
            first_iter = True

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)
                init_mat = np.copy(self.p_t_given_y)

                for i in range(0, cardinality_Y):
                    old_cluster = np.argmax(self.p_t_given_y[i, :])

                    if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                        self.p_t_given_y[i, old_cluster] = 0
                        self.p_t_given_y[i, -1] = 1

                        cur_card_T += 1


                        # calculate p(t)  new
                        # only needs to be updated in the following columns: old_cluster, last cluster,
                        # previous new cluster
                        # special dot test



                        if first_iter:
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)
                            self.p_x_and_t = np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)
                            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * self.p_x_and_t
                            first_iter = True

                        else:
                            rel_vec = np.array([old_cluster, -1, ind_min])
                            # add mass in temp cluster and in previously chosen cluster
                            self.p_t[-1] = p_y[i]
                            self.p_t[ind_min] += p_y[i-1]
                            self.p_t[old_cluster] -= p_y[i]

                            #self.p_t[rel_vec] = (self.p_t_given_y[:, rel_vec] * p_y[:, np.newaxis]).sum(0)

                            self.p_x_and_t[-1, :] = self.p_x_y[i, :]
                            self.p_x_and_t[ind_min, :] += self.p_x_y[i - 1, :]
                            self.p_x_and_t[old_cluster, :] -= self.p_x_y[i, :]

                            #self.p_x_and_t[rel_vec] = np.dot( self.p_t_given_y[:, rel_vec].T, self.p_x_y)
                            self.p_x_given_t[rel_vec] = 1 / (self.p_t[rel_vec, np.newaxis]) * self.p_x_and_t[rel_vec, :]

                        #ind_min, costs = inf_c.calc_merger_cost_sIB(self.p_t, self.p_x_given_t)
                        #merger_costs_vec_c = inf_c.calc_merger_cost_sIB(self.p_t, self.p_x_given_t)
                        merger_costs_vec = self.calc_merger_cost()
                        #if np.linalg.norm(merger_costs_vec-merger_costs_vec_c) >1e-7:
                        #    raise RuntimeError('Kacke')

                        ind_min = np.argmin(merger_costs_vec)
                        self.p_t_given_y[i, ind_min] = 1
                        self.p_t_given_y[i, -1] = 0
                        cur_card_T -= 1

                end_mat = self.p_t_given_y

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

            p_t_given_y_mats[:, :, run] = self.p_t_given_y
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t

            p_ty = self.p_t_given_y[:, :self.cardinality_T] * p_y[:, np.newaxis]
            p_xt = self.p_x_given_t[:self.cardinality_T,:] * self.p_t[:, np.newaxis]

            I_YT[run] = inf_tool.mutual_information(p_ty)
            I_TX[run] = inf_tool.mutual_information(p_xt)

            ib_fct[run] = I_YT[run] / (-self.beta) + I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = inf_tool.mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]

