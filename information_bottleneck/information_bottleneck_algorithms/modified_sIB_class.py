import numpy as np

from information_bottleneck.information_bottleneck_algorithms.generic_IB_class import GenericIB
from information_bottleneck.tools import inf_theory_tools as inf_tool

__author__ = "Maximilian Stark"
__copyright__ = "22.05.2016, Institute of Communications, Hamburg University of Technology"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__ = """This module contains a modified version of the original sequential Information Bottleneck
            The first important distinction to the classical sequential Information Bottleneck algorithm is, that the
            algorithm does start with a random initial clustering, but all clusters are forced to consist just of
            sequentially enumerated members y \in Y. Such a clustering can easily be constructed by separating a linear
            vector of sequent integers from 0 to |Y| into different sub vectors. During the rest of this algorithm,
            we aim to keep this natural order in the clusters and as a consequence do not run a draw-and-merge loop over
            all y \in Y, but instead only over the elements which are neighboured to one of the |T | cluster
            borders. As we want to keep the natural order of our initial clustering, we only consider two possible
            merging possibilities, where the one is putting a drawn element back to its original cluster and the other
            one is putting it to the neighboured cluster."""


def bool2int(x):
    y = 0
    for i, j in enumerate(x.astype(int)):
        y += j<<i
    return y

class modified_sIB(GenericIB):
    """This class realizes the calculation of the modified sequential Information Bottleneck algorithm.
      Description:
        The first important distinction to the classical sequential Information Bottleneck algorithm is, that the
        algorithm does start with a random initial clustering, but all clusters are forced to consist just of
        sequentially enumerated members y \in Y. Such a clustering can easily be constructed by separating a linear
        vector of sequent integers from 0 to |Y| into different sub vectors. During the rest of this algorithm,
        we aim to keep this natural order in the clusters and as a consequence do not run a draw-and-merge loop over
        all y \in Y, but instead only over the elements which are neighboured to one of the |T | cluster
        borders. As we want to keep the natural order of our initial clustering, we only consider two possible
        merging possibilities, where the one is putting a drawn element back to its original cluster and the other
        one is putting it to the neighboured cluster.
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
        self.name = 'modified sIB'

    def run_IB_algo(self):
        self.modified_sIB_algo()

    def calc_merger_cost(self, border_between_clusters, cur_card_T_):
        """Return the merger cost for putting one event in a cluster. Since this a modified version of the sIB only two
            clusters have to be tested. Which constrains the calculation to two comparisons.
            Args:
                :param border_between_clusters: denotes the current border between two adjacent cluster, optimized in this step
                :param cur_card_T_: The current cardinality T, meaning the cluster size, which is increased during the
                                    algorithm due to temporary clusters
            Return
                :return cost_vec:
            """

        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element
        bbc = border_between_clusters
        cur_card_T = cur_card_T_-1
        p_t_bar = self.p_t[cur_card_T] + self.p_t[[bbc, bbc+1]]

        pi1 = self.p_t[cur_card_T] / p_t_bar
        pi2 = self.p_t[[bbc, bbc+1]] / p_t_bar

        cost_vec = p_t_bar * (inf_tool.js_divergence(self.p_x_given_t[cur_card_T, :], self.p_x_given_t[[bbc, bbc+1], :], pi1, pi2)
                              -(pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)

        return cost_vec

    def modified_sIB_algo(self):
        """ The modified algorithm only optimizes over adjacent clusters to maintain the natural order of the initial
        event space"""

        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X=p_x.shape[0]
        cardinality_Y=p_y.shape[0]

        cur_card_T = self.cardinality_T

        # Initialization
        # number of identity matrices fitting inside p_t_givem_y
        neye = np.floor(cardinality_Y / (self.cardinality_T + 1))
        # neye = np.floor(cardinality_Y / (self.cardinality_T))

        # remaining rows that will be filled up with ones in the first row
        remainder = (cardinality_Y - neye * self.cardinality_T)

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_YT = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.cardinality_T + 1, self.nror))
        p_t_mats = np.zeros((1, self.cardinality_T, self.nror))
        p_x_given_t_mats = np.zeros((self.cardinality_T, cardinality_X, self.nror))

        # run for-loop for each number of run
        for run in range(0, self.nror):
            self.printProgress(run, self.nror, prefix='Run:')
            self.p_t_given_y = np.zeros((cardinality_Y, self.cardinality_T + 1))

            # Use dirichlet distribution to sample a valid density
            # the ones vector indicate that The expected value for each cluster is 1/(cardinaltiy_T)
            # Multiplying with constant changes variance. 1 means very high variance creating asymmetric clusters.
            # In the end +1 is needed to ensure that
            alpha = np.ones(int(self.cardinality_T)) * 1
            border_vec = np.ones(alpha.shape[0]) * cardinality_Y
            while border_vec[:-1].cumsum().max() >= cardinality_Y :
                border_vec = np.floor(np.random.dirichlet(alpha, 1).transpose() * (cardinality_Y ))
                border_vec[border_vec == 0] = 1

            border_vec = np.hstack([border_vec[:-1].cumsum(), cardinality_Y ]).astype(np.int)

            a = 0
            for t in range(0, self.cardinality_T ):
                self.p_t_given_y[a:border_vec[t], t] = 1
                a = border_vec[t]

            # Processing
            init_mat = self.p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.cardinality_T + 1))

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)
                last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 1])
                init_mat = np.copy(self.p_t_given_y)

                for border_between_clusters in range(0,self.cardinality_T - 1):
                    done_left_to_right = False
                    done_right_to_left = False

                    while not done_left_to_right:
                        done_left_to_right = True

                        # find last element in the cluster
                        # this is a trick here because argmax returns first hit so we flip the array first.
                        last_elem = self.p_t_given_y.shape[0] - np.argmax(self.p_t_given_y[::-1, border_between_clusters] > 0) - 1

                        old_cluster = border_between_clusters

                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[last_elem, :] = last_cluster_vec
                            cur_card_T += 1

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1/(self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[last_elem, border_between_clusters] = 1
                                #print('stop')
                            else:
                                self.p_t_given_y[last_elem, border_between_clusters+1] = 1
                                done_left_to_right = False

                            self.p_t_given_y[last_elem, cur_card_T-1] = 0
                            cur_card_T -= 1

                    # check other direction
                    while not done_right_to_left:
                        done_right_to_left = True

                        # find first element in the cluster
                        first_elem = np.argmax(self.p_t_given_y[:, border_between_clusters + 1] > 0)

                        old_cluster = border_between_clusters + 1

                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[first_elem, :] = last_cluster_vec
                            cur_card_T += 1

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[first_elem, border_between_clusters] = 1
                                done_right_to_left = False
                            else:
                                self.p_t_given_y[first_elem, border_between_clusters + 1] = 1

                            self.p_t_given_y[first_elem, cur_card_T-1] = 0
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
