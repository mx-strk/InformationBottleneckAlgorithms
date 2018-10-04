import numpy as np

from information_bottleneck.information_bottleneck_algorithms.generic_IB_class import GenericIB
from information_bottleneck.tools import inf_theory_tools as inf_tool

__author__ = "Maximilian Stark"
__copyright__ = "22.05.2016, Institute of Communications,Hamburg University of Technology "
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__="""This module contains the Agglomerative Information Bottleneck"""


class aIB(GenericIB):
    """This class can be used to perform the Agglomerative Information Bottleneck algorithm.
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

    def __init__(self, p_x_y_, card_T_):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], 5)
        self.name = 'agglomerative IB'

        self.cardinality_X=p_x_y_.shape[1]
        self.cardinality_Y=p_x_y_.shape[0]
        self.cost_mat= np.zeros((self.cardinality_Y,self.cardinality_Y))+np.inf

        self.merge_costs = dict()

        self.index_list = np.arange(self.cardinality_Y).tolist()
        self.index_vec = np.arange(self.cardinality_Y)

    def calc_merger_cost_pair(self,ind1,ind2):
        """Return the merger cost for putting one event in a cluster.
        Args:
            p_t: is a 1 x card_T array
            p_x_given_t: is a card_X x card_T array
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element

        ind1 = self.index_list.index(ind1)
        ind2 = self.index_list.index(ind2)

        p_t_bar = self.p_t[ind1] + self.p_t[ind2]

        pi1 = self.p_t[ind1] / p_t_bar
        pi2 = self.p_t[ind2] / p_t_bar

        self.cost_mat[ind1,ind2] = p_t_bar * (inf_tool.js_divergence(self.p_x_given_t[ind1, :], self.p_x_given_t[ind2, :], pi1, pi2) -
                              (pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)


        if self.cost_mat[ind1,ind2]<0:
            self.cost_mat[ind1, ind2] = 0

        return self.cost_mat[ind1,ind2]

    def calc_merger_cost_pair_vec(self,ind1,ind2):
        """Return the merger cost for putting one event in a cluster.
        Args:
            p_t: is a 1 x card_T array
            p_x_given_t: is a card_X x card_T array
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element

        p_t_bar = self.p_t[ind1] + self.p_t[ind2]

        pi1 = self.p_t[ind1] / p_t_bar
        pi2 = self.p_t[ind2] / p_t_bar

        self.cost_mat[ind1,ind2] = p_t_bar * (inf_tool.js_divergence(self.p_x_given_t[ind1, :], self.p_x_given_t[ind2, :], pi1, pi2) -
                              (pi1 * np.log2(pi1) + pi2 * np.log2(pi2)) / self.beta)

        self.cost_mat[self.cost_mat < 0] = 0

    def calc_all_merge_costs(self):
        """
        This function is called only once, during initialization of Partition
        Subsequent calls operate on a subset of the data
        """

        ind1 = np.kron(np.arange(self.cardinality_Y), np.ones(self.cardinality_Y))
        ind2 = np.tile(np.arange(self.cardinality_Y),self.cardinality_Y)
        valid_combinations = ind1<ind2
        ind1 = ind1[valid_combinations].astype(int)
        ind2 = ind2[valid_combinations].astype(int)

        self.calc_merger_cost_pair_vec(ind1, ind2)


    def find_merge_pair(self):
        """
        Search all cluster pairs for the best pair to merge.
        Use the following criteria:
        1) Find pair(s) for which merge cost is minimized
        2) If multiple candidates from (1), find pair with smallest inter-cluster distance
        """

        min_pair = min(self.merge_costs, key=lambda x: self.merge_costs[x])

        min_val = self.merge_costs[min_pair]

        assert min_val == self.calc_merger_cost_pair(*min_pair)
        ties = [k for k, v in self.merge_costs.items() if v == min_val]

        if len(ties) > 1:
            min_pair = ties[0]
        return min_pair

    def find_merge_pair_vec(self):
        """
        Search all cluster pairs for the best pair to merge.
        Use the following criteria:
        1) Find pair(s) for which merge cost is minimized
        2) If multiple candidates from (1), find pair with smallest inter-cluster distance
        """

        min_pair = np.unravel_index(np.argmin(self.cost_mat), self.cost_mat.shape)

        return min_pair

    def merge(self,i,j):

        target, remove = sorted([i, j])

        target = self.index_list.index(target)
        remove = self.index_list.index(remove)

        del self.index_list[remove]

        # delete column in p(t|y)
        self.p_t_given_y[:,target] = self.p_t_given_y[:,target]+self.p_t_given_y[:,remove]
        self.p_t_given_y = np.delete(self.p_t_given_y, remove, axis=1)

        # delete row in p(x|t)
        # update p(t)

        self.p_t[target] = self.p_t[target] + self.p_t[remove]
        self.p_t = np.delete(self.p_t, remove, axis=0)

        self.p_x_and_t[target,:] = self.p_x_and_t[target,:] + self.p_x_and_t[remove,:]
        self.p_x_and_t = np.delete(self.p_x_and_t, remove, axis=0)

        self.p_x_given_t[target,:] = 1 / self.p_t[target] * (self.p_x_and_t[target,:] )
        self.p_x_given_t = np.delete(self.p_x_given_t, remove, axis=0)

    def merge_vec(self,i,j):

        target, remove = sorted([i, j])

        del self.index_list[remove]

        # delete column in p(t|y)
        self.p_t_given_y[:,target] = self.p_t_given_y[:,target]+self.p_t_given_y[:,remove]
        self.p_t_given_y = np.delete(self.p_t_given_y, remove, axis=1)

        # delete row in p(x|t)
        # update p(t)

        self.p_t[target] = self.p_t[target] + self.p_t[remove]
        self.p_t = np.delete(self.p_t, remove, axis=0)

        self.p_x_and_t[target,:] = self.p_x_and_t[target,:] + self.p_x_and_t[remove,:]
        self.p_x_and_t = np.delete(self.p_x_and_t, remove, axis=0)

        self.p_x_given_t[target,:] = 1 / self.p_t[target] * (self.p_x_and_t[target,:] )
        self.p_x_given_t = np.delete(self.p_x_given_t, remove, axis=0)

    def merge_next(self):
        """
        Iterate the AIB algorithm.
        Find best pair to merge, perform merge, update clusters and merge costs for next iteration
        """
        # Decide which pair of clusters to merge next
        min_pair = self.find_merge_pair_vec()

        # Execute merge
        self.merge_vec(*min_pair)
        """After merge, recompute costs related to the merged clusters
        Two steps:
            1) Update pointers to point to the merged pair (the min of min_pair)
            2) Process this list with clusters.calc_merge_cost
        """

        target, remove = sorted(min_pair)
        # entries are basically the row and column of remove
        self.cost_mat = np.delete(self.cost_mat, remove,axis=0)
        self.cost_mat = np.delete(self.cost_mat, remove,axis=1)

        #entries to update
        # all entries in target row and column that are not inf
        dummy_vec=np.arange(self.cost_mat.shape[0])
        # check column entries, i.e. ind1 is fixed and ind2 is determined by relevant_entries
        relevant_entries = np.logical_not( np.isinf(self.cost_mat[target,:]))
        ind2 = dummy_vec[relevant_entries]
        ind1 = target * np.ones(ind2.shape[0])
        self.calc_merger_cost_pair_vec(ind1.astype(int),ind2.astype(int))

        # check row entries, i.e. ind2 is fixed and ind1 is determined by relevant_entries
        relevant_entries = np.logical_not( np.isinf(self.cost_mat[:,target]))
        ind1 = dummy_vec[relevant_entries]
        ind2 = target * np.ones(ind1.shape[0])
        self.calc_merger_cost_pair_vec(ind1.astype(int), ind2.astype(int))

    def run_IB_algo(self):
        self.aIB_algo()

    def aIB_algo(self):
        """ This function tries to minimize the information bottleneck functional using the Agglomerative IB (Agg-IB) algorithm.
        This algorithm only allows for deterministic cluster mapping, meaning beta is always set to infinity."""

        # set static values
        p_x = self.p_x_y.sum(0)
        p_y = self.p_x_y.sum(1)

        self.I_XY_list = []
        self.I_XT_list = []

        self.p_x = self.p_x_y.sum(0)
        self.p_y = self.p_x_y.sum(1)

        # in the first step T is a perfect copy of y.
        self.p_t = p_y.copy()
        self.p_t_shortened = self.p_t.copy()

        p_x_and_t =  self.p_x_y.copy()
        self.p_x_and_t = self.p_x_y.copy()

        self.p_x_given_t = p_x_and_t / self.p_t[:,np.newaxis]
        self.p_x_given_t_shortened = self.p_x_given_t.copy()

        self.p_t_given_y = np.eye(self.cardinality_Y)


        self.calc_all_merge_costs()

        while self.p_x_given_t.shape[0]>self.cardinality_T:

            self.merge_next()

            p_xt = self.p_x_given_t * self.p_t[:, np.newaxis]

            self.I_XY_list.append(inf_tool.mutual_information(self.p_x_y))
            self.I_XT_list.append(inf_tool.mutual_information(p_xt))

        self.MI_XY = inf_tool.mutual_information(self.p_x_y)
        self.MI_XT = inf_tool.mutual_information(p_xt)

