import numpy as np

from information_bottleneck.information_bottleneck_algorithms.generic_IB_class import GenericIB
from information_bottleneck.tools import inf_theory_tools as inf_tool

__author__ = "Maximilian Stark"
__copyright__ = "26.05.2016, Institute of Communications, Hamburg University of Technology"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__ = """The class implements the algorithm for the creation of the Trellis
        message mappings from the paper by J. Lewandowsky "Trellis based Node
        Operations for LDPC Decoders from the Information Bottleneck Method".
        It is an extended version of the modified sequential IB algorithm that additionally enforces symmetry of
        the clusters."""



class symmetric_sIB(GenericIB):
    """This class realizes the calculation of the modified sequential Information Bottleneck algorithm that outputs
        symmetric clusters.
      Description:
        The class implements the algorithm for the creation of the Trellis
        message mappings from the paper. It is an extended version of the
        modified sequential IB algorithm that additionally enforces symmetry of
        the clusters.
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
        self.name = 'symmetric sIB'

    def run_IB_algo(self):
        if self.cardinality_T == 2:
            self.cardinality_Y = self.p_x_y.shape[0]
            self.p_t_given_y = np.zeros((self.cardinality_Y, 2))
            self.p_t_given_y[0:int(self.cardinality_Y / 2) , 1] = 1
            self.p_t_given_y[int(self.cardinality_Y / 2):, 0] = 1
            self.p_y=self.p_x_y.sum(axis=1)

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :self.cardinality_T] * self.p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:self.cardinality_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :self.cardinality_T].T,
                                                                            self.p_x_y)
        else:
            self.symmetric_sIB_algo()

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
        cur_card_T = cur_card_T_-2
        p_t_bar = self.p_t[cur_card_T] + self.p_t[[bbc, bbc+1]]

        pi1 = self.p_t[cur_card_T] / p_t_bar
        pi2 = self.p_t[[bbc, bbc+1]] / p_t_bar

        cost_vec = p_t_bar * (inf_tool.js_divergence(self.p_x_given_t[cur_card_T, :], self.p_x_given_t[[bbc, bbc+1], :], pi1, pi2)
                              -(pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)

        return cost_vec

    def symmetric_sIB_algo(self):
        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        cardinality_X=p_x.shape[0]
        cardinality_Y=p_y.shape[0]

        cur_card_T = self.cardinality_T

        # Initialization

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_YT = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        p_t_given_y_mats = np.zeros((cardinality_Y, self.cardinality_T , self.nror))
        p_t_mats = np.zeros((1, self.cardinality_T, self.nror))
        p_x_given_t_mats = np.zeros((self.cardinality_T, cardinality_X, self.nror))
        all_needed = 0
        # run for-loop for each number of run
        for run in range(0, self.nror):
            self.printProgress(run, self.nror,prefix='Run:')

            self.p_t_given_y = np.zeros((cardinality_Y, self.cardinality_T + 2))

            # Initialization of p_t_given_y
            # Use dirichlet distribution to sample a valid density
            # the ones vector indicate that The expected value for each cluster is 1/(cardinaltiy_T)
            # Multiplying with constant changes variance. 1 means very high variance creating asymmetric clusters.
            # In the end +1 is needed to ensure that
            alpha = np.ones(int(self.cardinality_T / 2)) * 1
            border_vec = np.ones(alpha.shape[0]) * cardinality_Y / 2
            while border_vec[:-1].cumsum().max() >= cardinality_Y / 2:
                border_vec = np.floor(np.random.dirichlet(alpha, 1).transpose() * (cardinality_Y / 2))
                border_vec[border_vec == 0] = 1

            border_vec = np.hstack([border_vec[:-1].cumsum(), cardinality_Y / 2]).astype(int)

            a = 0
            for t in range(0, int(self.cardinality_T / 2)):
                self.p_t_given_y[a:border_vec[t], t] = 1
                self.p_t_given_y[cardinality_Y - border_vec[t] :cardinality_Y - a, self.cardinality_T - t -1] = 1
                a = border_vec[t]


            # Processing
            init_mat = self.p_t_given_y
            end_mat = np.zeros((cardinality_Y, self.cardinality_T + 2))

            # repeat until stable solution found
            while not np.array_equal(init_mat, end_mat):
                self.p_t = (self.p_t_given_y * p_y[:, np.newaxis]).sum(0)

                all_needed +=1

                last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 1, 0])
                partner_last_cluster_vec = np.hstack([np.zeros(self.cardinality_T), 0, 1])

                init_mat = np.copy(self.p_t_given_y)

                for border_between_clusters in range(0,int(self.cardinality_T/2) - 1):
                    done_left_to_right = False
                    done_right_to_left = False

                    while not done_left_to_right:
                        done_left_to_right = True
                        all_needed += 1
                        # find last element in the cluster
                        # this is a trick here because argmax returns first hit so we flip the array first.
                        last_elem = self.p_t_given_y.shape[0] - np.argmax(self.p_t_given_y[::-1, border_between_clusters] > 0) - 1

                        old_cluster = border_between_clusters

                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[last_elem, :] = last_cluster_vec
                            self.p_t_given_y[-(last_elem+1), :] = partner_last_cluster_vec

                            cur_card_T += 2

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1/(self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[last_elem, border_between_clusters] = 1
                                self.p_t_given_y[cardinality_Y-last_elem-1, cur_card_T-2-border_between_clusters-1] = 1

                            else:
                                self.p_t_given_y[last_elem, border_between_clusters+1] = 1
                                self.p_t_given_y[- (last_elem+1), cur_card_T - 2 - border_between_clusters-2] = 1
                                done_left_to_right = False

                            self.p_t_given_y[-(last_elem+1), -1] = 0
                            self.p_t_given_y[last_elem, - 2] = 0

                            cur_card_T -= 2

                    # check other direction
                    while not done_right_to_left:
                        done_right_to_left = True
                        all_needed += 1
                        # find first element in the cluster
                        first_elem = np.argmax(self.p_t_given_y[:, border_between_clusters + 1] > 0)

                        old_cluster = border_between_clusters + 1
                        if np.sum(self.p_t_given_y[:, old_cluster]) > 1:
                            self.p_t_given_y[first_elem, :] = last_cluster_vec
                            self.p_t_given_y[cardinality_Y - first_elem - 1, :] = partner_last_cluster_vec

                            cur_card_T += 2

                            # calculate p(t)  new
                            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

                            # calculate p(x | t) new
                            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

                            merger_costs_vec = self.calc_merger_cost(border_between_clusters, cur_card_T)

                            ind_min = np.argmin(merger_costs_vec)

                            if ind_min == 0:
                                self.p_t_given_y[first_elem, border_between_clusters] = 1
                                self.p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 2 - border_between_clusters-1] = 1

                                done_right_to_left = False
                            else:
                                self.p_t_given_y[first_elem, border_between_clusters + 1] = 1
                                self.p_t_given_y[cardinality_Y - first_elem-1, cur_card_T - 2 - border_between_clusters-2] = 1

                            self.p_t_given_y[cardinality_Y - first_elem - 1, cur_card_T - 1] = 0
                            self.p_t_given_y[first_elem, cur_card_T - 2] = 0
                            cur_card_T -= 2

                end_mat = self.p_t_given_y

            # calculate p(t)  new
            self.p_t = (self.p_t_given_y[:, :cur_card_T] * p_y[:, np.newaxis]).sum(0)

            # calculate p(x | t) new
            self.p_x_given_t = 1 / (self.p_t[:cur_card_T, np.newaxis]) * np.dot(self.p_t_given_y[:, :cur_card_T].T, self.p_x_y)

            p_t_given_y_mats[:, :, run] = self.p_t_given_y[:, :self.cardinality_T]
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t[:self.cardinality_T, :]

            p_ty = self.p_t_given_y[:, :self.cardinality_T] * p_y[:, np.newaxis]
            p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]

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