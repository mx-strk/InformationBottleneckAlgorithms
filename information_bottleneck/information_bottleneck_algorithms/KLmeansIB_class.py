import os

import numpy as np
import progressbar
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools as cl_tools
from mako.template import Template

from information_bottleneck.information_bottleneck_algorithms.generic_IB_class import GenericIB
from information_bottleneck.tools import inf_theory_tools as inf_tool

__author__ = "Maximilian Stark"
__copyright__ = "04.10.2018, Institute of Communications, Hamburg University of Technology "
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Release"
__doc__="This module contains the KL Means Algorithm"


class KLmeansIB(GenericIB):
    """This class can be used to perform the KL-Means Information Bottleneck algorithm.
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
          p_x_and_t
          p_t
      Note: The values can be accessed using the appropriate return functions of the class
        self.return_results         return all values
        self.display_MI             return the MIs graphically as well as percentage of maintained mutual information
      """

    def __init__(self, p_x_y_, card_T_,symmetric_init = False ,nror=50):
        GenericIB.__init__(self, p_x_y_, card_T_, np.inf, [], nror)
        self.name = 'KL means IB'

        self.cardinality_X=p_x_y_.shape[1]
        self.cardinality_Y=p_x_y_.shape[0]
        #self.cost_mat= np.zeros((self.cardinality_Y,self.cardinality_Y))+np.inf

        self.symmetric_init = symmetric_init
        self.p_y = self.p_x_y.sum(1)
        self.p_t = np.zeros(self.cardinality_T)

        # calculate p(x|y)
        self.p_x_given_y = self.p_x_y / self.p_y[:, np.newaxis]

        # sort events by gravity
        #self.sort_events_by_cog_idcs()


    def kl_divergence_mat_col(self,pdf1, pdf2):
        KL_div = np.zeros((pdf1.shape[0], pdf2.shape[0]))

        for c in range(pdf2.shape[0]):
            KL_div[:, c] = inf_tool.kl_divergence(self.p_x_given_y, self.p_x_given_t[c,:])

        return KL_div

    def estimate(self):
        p_t_given_y = np.argmin(self.kl_divergence_mat_col(self.p_x_given_y, self.p_x_given_t), axis=1)

        #ensure that no cluster is empty
        for t in range(self.cardinality_T):
            indices = np.where(p_t_given_y == t)[0]
            if indices.size == 0:
                indices = self.last_resort[t]
                p_t_given_y[int(indices)] = int(t)
            else:
                self.last_resort[t] = indices[-1]

        return p_t_given_y

    def init_openCL(self,set_mem_pool_None = False):
        self.context = cl.create_some_context()
        print('###  OPENCL Device #####')
        print(self.context.get_info(cl.context_info.DEVICES))

        path = os.path.split(os.path.abspath(__file__))
        kernelsource = open(os.path.join(path[0], "IB_kernels.cl")).read()
        tpl = Template(kernelsource)
        rendered_tp = tpl.render(cardinality_T=self.cardinality_T)


        #kernelsource = open("information_bottleneck / information_bottleneck_algorithms / IB_kernels.cl").read()

        self.program = cl.Program(self.context, str(rendered_tp)).build()
        self.queue = cl.CommandQueue(self.context)
        if set_mem_pool_None:
            self.mem_pool = None
        else:
            self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))

        self.p_x_given_y_buffer = cl_array.to_device(self.queue, self.p_x_given_y.astype(dtype=np.float64),allocator=self.mem_pool)
        self.p_x_and_y_buffer = cl_array.to_device(self.queue, self.p_x_y.astype(dtype=np.float64),allocator=self.mem_pool)
        self.p_y_buffer = cl_array.to_device(self.queue, self.p_y.astype(dtype=np.float64),allocator=self.mem_pool)


        self.p_x_and_t_buffer = cl_array.empty(self.queue, (self.cardinality_T, self.cardinality_X), dtype=np.float64,
                                               allocator=self.mem_pool)
        self.p_t_buffer = cl_array.empty(self.queue, self.cardinality_T, dtype=np.float64,
                                               allocator=self.mem_pool)
        self.argmin_buffer = cl_array.empty(self.queue,self.cardinality_Y,dtype=np.int32,allocator=self.mem_pool)
        self.dkl_mat_buffer = cl_array.empty(self.queue,(self.cardinality_Y,self.cardinality_T),dtype=np.float64,allocator=self.mem_pool)
        self.start_vec_buffer = cl_array.empty(self.queue,self.cardinality_T,dtype=np.int32,allocator=self.mem_pool)



        self.dkl_compute_prog = self.program.compute_dkl_mat
        self.dkl_compute_prog.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])

        self.find_argmin_prog = self.program.find_argmin
        self.find_argmin_prog.set_scalar_arg_dtypes([np.int32, np.int32, None, None])



        self.allow_move_prog = self.program.allow_move
        self.allow_move_prog.set_scalar_arg_dtypes([np.int32, None, None, None])

        self.compute_p_x_and_t_parallel_prog = self.program.compute_p_x_and_t_parallel
        self.compute_p_x_and_t_parallel_prog.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None, None, None])


        self.compute_p_x_given_t_parallel_prog = self.program.compute_p_x_given_t_parallel
        self.compute_p_x_given_t_parallel_prog.set_scalar_arg_dtypes(
            [np.int32, None, None])

        self.compute_p_t_parallel_prog = self.program.compute_p_t_parallel
        self.compute_p_t_parallel_prog.set_scalar_arg_dtypes([np.int32, None, None])


        self.update_dist_prog = self.program.update_distributions
        self.update_dist_prog.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None,None, None])

    def opencl_cleanup(self):
        if self.mem_pool != None:
            self.mem_pool.stop_holding()
            del self.mem_pool

        del self.program
        del self.queue
        del self.context

        del self.p_x_given_y_buffer
        del self.p_x_and_y_buffer
        del self.p_y_buffer
        #del self.p_t_given_y_buffer
        #del self.p_x_given_t_buffer
        del self.p_x_and_t_buffer
        del self.p_t_buffer
        del self.dkl_mat_buffer

        del self.dkl_compute_prog
        del self.allow_move_prog
        del self.update_dist_prog

        pass

    def estimate_opencl(self, to_devive = False, to_host = False):

        if to_devive:
            self.p_x_given_t_buffer = cl_array.to_device(self.queue, self.p_x_given_t,allocator=self.mem_pool)

        self.dkl_compute_prog(self.queue, (self.cardinality_Y, self.cardinality_T), None,
                              self.cardinality_T,
                              self.cardinality_Y,
                              self.cardinality_X,
                              self.p_x_given_t_buffer.data,
                              self.p_x_given_y_buffer.data,
                              self.dkl_mat_buffer.data)

        self.queue.finish()
        self.find_argmin_prog(self.queue, (self.cardinality_Y,),
                              None,
                              self.cardinality_T,
                              self.cardinality_Y,
                              self.dkl_mat_buffer.data,
                              self.argmin_buffer.data
                              )
        self.queue.finish()
        self.allow_move_prog(self.queue, (1,),
                             None,
                             self.cardinality_Y,
                             self.argmin_buffer.data,
                             self.p_t_given_y_buffer.data,
                             self.length_vec_buffer.data
                             )
        self.queue.finish()
        if to_host:
            return self.p_t_given_y_buffer.get()

    def update_distributions_opencl(self):
        ## to optimize the structure of the marginalization we extract the locations of events for a cluster in adavance
        self.start_vec = np.hstack((0, np.cumsum(self.length_vec_buffer.get())[:-1] ))
        self.start_vec_buffer = cl_array.to_device(self.queue, self.start_vec.astype(dtype=np.int32), allocator = self.mem_pool)

        # read p_t_given_y_buffer and perform argsort
        self.ordered_cluster_location_vec = np.argsort(self.p_t_given_y_buffer.get())
        self.ordered_cluster_location_vec_buffer = cl_array.to_device(self.queue, self.ordered_cluster_location_vec.astype(dtype=np.int32), allocator = self.mem_pool)

        self.compute_p_x_and_t_parallel_prog(self.queue, (self.cardinality_T, self.cardinality_X),
                                             None,
                                             self.cardinality_T,
                                             self.cardinality_Y,
                                             self.cardinality_X,
                                             self.p_x_and_t_buffer.data,
                                             self.p_x_and_y_buffer.data,
                                             self.ordered_cluster_location_vec_buffer.data,
                                             self.start_vec_buffer.data,
                                             self.length_vec_buffer.data)

        self.queue.finish()

        self.compute_p_t_parallel_prog(self.queue, (self.cardinality_T, ),
                                     None,
                                     self.cardinality_X,
                                     self.p_x_and_t_buffer.data,
                                     self.p_t_buffer.data)

        self.queue.finish()
        # please be aware that self.p_x_and_t_buffer is changed, i.e., at the end p_x_and_t_buffer will hold
        # p_x_and_t
        self.compute_p_x_given_t_parallel_prog(self.queue, (self.cardinality_T, self.cardinality_X ),
                                     None,
                                     self.cardinality_X,
                                     self.p_x_and_t_buffer.data,
                                     self.p_t_buffer.data)

        self.p_x_given_t_buffer = self.p_x_and_t_buffer

        self.queue.finish()

        return self.p_t_buffer.get(), self.p_x_given_t_buffer.get()

    def run_IB_algo(self,set_mem_pool_None = False):
        self.KL_means_algo_opencl(set_mem_pool_None = set_mem_pool_None)
        self.sort_clusters_by_cog_idcs()

    def KL_means_algo_opencl(self, set_mem_pool_None = False):
        """ This function tries to minimize the information bottleneck functional using a KL means_algorithm."""
        self.init_openCL(set_mem_pool_None)

        # Initialization
        # number of identity matrices fitting inside p_t_givem_y
        neye = int(np.floor(self.cardinality_Y / (self.cardinality_T )))
        # remaining rows that will be filled up with ones in the first row
        remainder = int((self.cardinality_Y - neye * self.cardinality_T))

        # preallocate arrays
        ib_fct = np.zeros(self.nror)
        I_TX = np.zeros(self.nror)
        counter_vec = np.zeros(self.nror)
        self.I_TX_evolution_list = [ [0] ] * self.nror

        I_TX_winner = 0

        # run for-loop for each number of run
        bar = progressbar.ProgressBar(widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])

        for run in bar(range(0, self.nror)):

            # Begin initialization
            if self.symmetric_init:
                self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T))
                # Initialization of p_t_given_y
                # Use dirichlet distribution to sample a valid density
                # the ones vector indicate that The expected value for each cluster is 1/(cardinaltiy_T)
                # Multiplying with constant changes variance. 1 means very high variance creating asymmetric clusters.
                # In the end +1 is needed to ensure that
                alpha = np.ones(int(self.cardinality_T / 2)) * 100
                border_vec = np.ones(alpha.shape[0]) * self.cardinality_Y / 2
                while border_vec[:-1].cumsum().max() >= self.cardinality_Y / 2:
                    border_vec = np.floor(np.random.dirichlet(0.1 * alpha, 1).transpose() * (self.cardinality_Y / 2))
                    border_vec[border_vec == 0] = 1

                border_vec = np.hstack([border_vec[:-1].cumsum(), self.cardinality_Y / 2]).astype(int)

                a = 0
                for t in range(0, int(self.cardinality_T / 2)):
                    self.p_t_given_y[a:border_vec[t], t] = 1
                    self.p_t_given_y[self.cardinality_Y - border_vec[t]:self.cardinality_Y - a,
                    self.cardinality_T - t - 1] = 1
                    a = border_vec[t]

            else:
                self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T + 1))
                self.p_t_given_y[:int(neye * self.cardinality_T), :self.cardinality_T] = np.tile(
                    np.eye(self.cardinality_T),
                    (neye, 1))
                self.p_t_given_y[self.cardinality_Y - remainder:, 0] = np.ones(remainder)

                self.p_t_given_y = self.p_t_given_y[np.random.permutation(self.cardinality_Y), :]

            p_t_given_y = np.argmax(self.p_t_given_y, axis=1)

            for t in range(self.cardinality_T):
                indices = np.where(p_t_given_y == t)[0]
                # calculate p(t)
                self.p_t[t] = self.p_y[indices].sum(0)
                # calculate p(x|t)
                self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)


            p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]
            old_MI = inf_tool.mutual_information(p_xt)
            self.I_TX_evolution_list[run] = [old_MI]
            new_MI = 0

            self.p_x_given_t_buffer = cl_array.to_device(self.queue, self.p_x_given_t.astype(dtype=np.float64),allocator=self.mem_pool)
            self.p_t_given_y_buffer = cl_array.to_device(self.queue, p_t_given_y.astype(dtype=np.int32),allocator=self.mem_pool)

            # Processing
            # the vector holds the number of elements per cluster
            length_vec = np.zeros(self.cardinality_T)
            for t in range(self.cardinality_T):
                length_vec[t] = np.sum(p_t_given_y==t)

            self.length_vec_buffer = cl_array.to_device(self.queue, length_vec.astype(dtype=np.int32),allocator=self.mem_pool)

            counter = 0
            # repeat until stable solution found
            while np.abs(old_MI - new_MI) > 1e-11 and counter < self.cardinality_T * 10:
                counter += 1
                old_MI = new_MI

                ### OPENCL Routine
                # estimation step
                self.estimate_opencl()
                # update step
                self.p_t, self.p_x_given_t = self.update_distributions_opencl()
                p_xt = self.p_x_given_t * self.p_t[:, np.newaxis]
                new_MI = inf_tool.mutual_information(p_xt)
                self.I_TX_evolution_list[run].append(new_MI)

            # load data from OpenCL device
            p_t_given_y = self.p_t_given_y_buffer.get()

            # free buffer
            del self.p_t_given_y_buffer
            del self.p_x_given_t_buffer
            del self.length_vec_buffer

            p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]
            p_xt = p_xt / p_xt.sum()

            I_TX[run] = inf_tool.mutual_information(p_xt)
            ib_fct[run] = I_TX[run]

            counter_vec[run] = counter

            if I_TX_winner < I_TX[run]:
                # check if new winner
                p_t_given_y_winner = p_t_given_y
                p_t_winner = self.p_t
                p_x_given_t_winner = self.p_x_given_t

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)

        # blow up p(t|y)
        self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T))
        self.p_t_given_y[np.arange(p_t_given_y_winner.shape[0]), p_t_given_y] = 1

        self.p_x_given_t = p_x_given_t_winner
        self.p_t = p_t_winner

        self.MI_XY = inf_tool.mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]

        self.opencl_cleanup()

    def sort_clusters_by_cog_idcs(self):
        cogs = np.dot(self.p_x_given_t,np.linspace(0, self.cardinality_X-1, self.cardinality_X)[:,np.newaxis])
        order_idcs = np.argsort(cogs.squeeze())
        
        self.p_x_given_t [:,:] = self.p_x_given_t [order_idcs,:]
        self.p_t [:] = self.p_t [order_idcs]
        self.p_t_given_y[:,:]=self.p_t_given_y[:,order_idcs]
        
        pass

    def sort_events_by_cog_idcs(self):
        # calculate p(x|y)

        self.input_dist = self.p_x_y.copy()

        self.cardinality_X = self.p_x_y.shape[1]
        self.cardinality_Y = self.p_x_y.shape[0]
        self.p_y = self.p_x_y.sum(1)
        #self.p_x_given_y = self.p_x_y / self.p_y[:, np.newaxis]
        self.p_y_given_x = self.p_x_y / self.p_x_y.sum(axis=0)


        index = []
        for x_dim in range(self.p_y_given_x.shape[1]):
            cur_set = np.argwhere(np.argmax(self.p_x_y, axis=1) == x_dim).squeeze()
            neg_log_like = -np.log(self.p_y_given_x[cur_set])

            sign_vec = 0
            if x_dim != 0:
                if x_dim != self.p_y_given_x.shape[1] - 1:
                    sorted_probs = np.argsort(self.p_x_y[cur_set, :])
                    # sign_vec = -1 * (neg_log_like[:,x_dim - 1] < neg_log_like[:,x_dim + 1])
                    sign_vec = -1 * (sorted_probs[:, 2] < sorted_probs[:, 3])
                else:
                    sign_vec = -1

            neg_log_like = neg_log_like[:, x_dim] * (2 * sign_vec + 1)

            ordered_idcs_set = np.argsort(neg_log_like)
            index = np.append(index, cur_set[ordered_idcs_set])

            order_idcs = index.astype(int)


        self.p_y_given_x[:, :] = self.p_y_given_x[order_idcs, :]
        self.p_y[:] = self.p_y[order_idcs]
        self.p_x_y[:] = self.p_x_y[order_idcs, :]

    def KL_means_algo(self):
        """ This function tries to minimize the information bottleneck functional using a KL means_algorithm."""

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

        # run for-loop for each number of run
        for run in range(0, self.nror):
            self.printProgress(run, self.nror, prefix='Run:')

            # Beginn initialization
            self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T + 1))
            self.p_t_given_y[:int(neye * self.cardinality_T), :self.cardinality_T] = np.tile(np.eye(self.cardinality_T),
                                                                                             (neye, 1))
            self.p_t_given_y[self.cardinality_Y - remainder:, 0] = np.ones(remainder)

            self.p_t_given_y = self.p_t_given_y[np.random.permutation(self.cardinality_Y), :]

            p_t_given_y= np.argmax(self.p_t_given_y,axis=1)

            self.last_resort = np.zeros(self.cardinality_T) # these vector has to ensure that at least one entry is in one cluster
            for t in range(self.cardinality_T):
                indices = np.where(p_t_given_y == t)[0]
                # grab one entry from each cluster
                smallest_contribution = np.argmin(self.p_y[indices])
                self.last_resort[t] = indices[smallest_contribution]
                # calculate p(t)
                self.p_t[t] = self.p_y[indices].sum(0)
                # calculate p(x|t)
                self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)


            p_xt = self.p_x_given_t[:self.cardinality_T, :] * self.p_t[:, np.newaxis]
            old_MI = inf_tool.mutual_information(p_xt)
            new_MI = 0

            # Processing
            counter = 0
            # repeat until stable solution found
            while np.abs(old_MI-new_MI)>1e-11 and counter<self.cardinality_T*10:
                counter += 1
                old_MI = new_MI

                # estimation step
                p_t_given_y = self.estimate()

                for t in range(self.cardinality_T):
                    indices = np.where(p_t_given_y == t)[0]
                    if indices.size==0:
                        indices = self.last_resort[t]
                        p_t_given_y[int(indices)] = int(t)
                        self.p_t[t] = self.p_y[int(indices)]
                        self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[int(indices), :]
                    else:
                        # grab one entry from each cluster
                        self.last_resort[t] = indices[-1]
                        # calculate p(t)
                        self.p_t[t] = self.p_y[indices].sum(0)
                        # calculate p(x|t)
                        self.p_x_given_t[t, :] = (1 / self.p_t[t]) * self.p_x_y[indices, :].sum(0)


                p_xt = self.p_x_given_t * self.p_t[:, np.newaxis]
                new_MI = inf_tool.mutual_information(p_xt)

            self.p_t_given_y = np.zeros((self.cardinality_Y, self.cardinality_T ))
            self.p_t_given_y[:,p_t_given_y] = 1
            counter_vec[run]=counter
            p_t_given_y_mats[:, :, run] = self.p_t_given_y
            p_t_mats[:, :, run] = self.p_t
            p_x_given_t_mats[:, :, run] = self.p_x_given_t

            p_xt = self.p_x_given_t[:self.cardinality_T,:] * self.p_t[:, np.newaxis]
            p_xt = p_xt/p_xt.sum()

            I_TX[run] = inf_tool.mutual_information(p_xt)
            ib_fct[run] = I_TX[run]

        # choose the run maximizing the Information Bottleneck functional
        winner = np.argmax(ib_fct)
        print('Winner finished in ',counter_vec[winner],' iterations.')
        print('Average number of iterations to finished:', np.mean(counter_vec),)
        self.p_t_given_y = p_t_given_y_mats[:, :, winner].squeeze()
        self.p_x_given_t = p_x_given_t_mats[:, :, winner].squeeze()
        self.p_t = p_t_mats[:, :, winner].squeeze()
        self.MI_XY = inf_tool.mutual_information(self.p_x_y)
        self.MI_XT = I_TX[winner]
