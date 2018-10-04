import sys

import numpy as np

from information_bottleneck.tools import inf_theory_tools as inf_tool

__author__ = "Maximilian Stark"
__copyright__ = "18.05.2016, Institute of Communications, Hamburg University of Technology"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__ = """This module contains the definition of the abstract information bottleneck class. From this class all other
          information bottleneck algorithms are inherited:
          """


class GenericIB:
    """Common base class for all Information Bottleneck classes
    Args:
    input parameter
        p_x_y                   input joint pdf, where x is the number of columns and y the number of rows
    IB related parameters
        cardinality_T
        beta
        eps
        nror
    mutual information
        MI_XT                   mutual information of output I(X;T)
        MI_XY                   mutual information of input I(X;Y)
    output PDF_s
        p_t_given_y
        p_x_given_t
        p_t
    """

    def __init__(self, p_x_y_, card_T_, beta_, eps_ , nror_):
        # initialize parameters
        self.p_x_y = p_x_y_
        self.cardinality_T = card_T_
        self.beta = beta_
        self.eps = eps_
        self.nror = nror_

        # initialize unused parameters
        self.MI_XT = 1
        self.MI_XY = 1
        self.p_t_given_y = np.zeros((self.p_x_y.shape[0], self.cardinality_T))
        self.p_x_given_t = np.zeros((self.cardinality_T, self.p_x_y.shape[1]))
        self.p_t = np.zeros((1,self.cardinality_T))
        self.name = 'GenericClass'

        if card_T_ >= self.p_x_y.shape[0]:
            raise RuntimeError('The number of desired clusters is larger/equal than the input cardinality |T|>=|Y| !!')

    def calc_merger_cost(self):
        """Return the merger cost for putting one event in a cluster.
        Args:
            p_t: is a 1 x card_T array
            p_x_given_t: is a card_X x card_T array
        """
        # p_t_bar is the sum of the last element, corresponding to cardinality T, and the vector except of the last
        # element
        p_t_bar = self.p_t[ -1] + self.p_t[:- 1]

        pi1 = self.p_t[-1] / p_t_bar
        pi2 = self.p_t[:-1] / p_t_bar

        cost_vec = p_t_bar * (inf_tool.js_divergence(self.p_x_given_t[-1, :], self.p_x_given_t[:-1, :], pi1, pi2) -
                    (pi1 * np.log2(pi1)+pi2 * np.log2(pi2)) / self.beta)

        return cost_vec

    def run_IB_algo(self):
        """only template that will be used by the specific implementations later."""
        pass

    def return_results(self):
        """Return all parameters generate by the Information Bottleneck as dictionary
        Return:
            :return p_t_given_y:
            :return p_x_given_t:
            :return p_t:
            :return MI_XT:
            :return MI_XY:
        Note: These are the strings of the dictionary elements.
        """
        return {'p_t_given_y': self.p_t_given_y,
                'p_x_given_t': self.p_x_given_t,
                'p_t': self.p_t,
                'MI_XT': self.MI_XT,
                'MI_XY': self.MI_XY }

    def display_MIs(self,short=False):
        """Return the Mutual information for the input pdf and after applying the IB as well as the ratio of both in a
        graphical way.
        Args:
            None
        Return:
            None
        """
        if short:
            print('MI_XT_s= ', str(self.MI_XT))
        else:
            print('----- Mutual Information Comp --- ')
            print('----- ', self.name, ' ------ ')
            print('MI_XT_s= ', str(self.MI_XT))
            print('MI_XY_s= ', str(self.MI_XY))
            print('ratio= ', str(self.MI_XT / self.MI_XY))

    def printProgress(self, iteration, total, prefix='', suffix='', decimals=2, barLength=70):
        """
        Call in a loop to create terminal progress bar
        Args:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : number of decimals in percent complete (Int)
            barLength   - Optional  : character length of bar (Int)
        """
        filledLength = int(round(barLength * (iteration + 1) / float(total)))
        percents = round(100.00 * ((iteration + 1) / float(total)), decimals)
        bar = '#' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r')
        sys.stdout.write('%s [%s] %s%s %s / %s   %s \r' % (prefix, bar, percents, '%', iteration + 1, total, suffix)),
        sys.stdout.flush()
        if iteration+1 == total:
            print("\n")

    def get_results(self):
        return self.p_t_given_y, self.p_x_given_t, self.p_t

    def get_mutual_inf(self):
        return self.MI_XT, self.MI_XY