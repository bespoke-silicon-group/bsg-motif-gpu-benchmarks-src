/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#define _CUDA_COMPILER_

#include "support/common.h"
#include <stdio.h>
#include <assert.h>

__device__ __noinline__ bool shift_to_global(int *shift, int *tail, int *tail_bin, int *q2, int *l_q2, int *count, int *reached) {
    const int WG_SIZE = blockDim.x;
	/////////////////////////////////////////////////////////
    // Compute size of the output and allocate space in the global queue
    __syncthreads();
    int size = *tail_bin;
    bool ret = *count;
    if(!ret)
    	return ret;
    int tid = atomicAdd(reached, 1); // Hacky. Needed for GPGPU-Sim
    if(tid == 0) {
        *shift = atomicAdd(&tail[0], min(size, W_QUEUE_SIZE));
    }
    __syncthreads();
    ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
    int local_shift = tid;
    while(local_shift < min(size, W_QUEUE_SIZE)) {
        q2[*shift + local_shift] = l_q2[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += *reached;
    }
    __syncthreads();
    if(tid == 0) {
    	*count = 0;
    	*tail_bin = 0;
    	*reached = 0;
    }
    __syncthreads();
    return ret;
    //////////////////////////////////////////////////////////////////////////
}

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void SSSP_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *gray_shade, int *iter, int LIMIT, const int CPU) {

    extern __shared__ int l_mem[];
    int* tail_bin = l_mem;
    int* l_q2 = (int*)&tail_bin[1];
    int* shift = (int*)&l_q2[W_QUEUE_SIZE];
    int* base = (int*)&shift[1];
    int* count = (int*)&base[1];
	__shared__ int reached;

    const int tid     = threadIdx.x;
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAXWG   = gridDim.x;
    const int WG_SIZE = blockDim.x;

    int iter_local = atomicAdd(&iter[0], 0);

    int n_t_local = atomicAdd(n_t, 0);

    int gray_shade_local = atomicAdd(&gray_shade[0], 0);

    if(tid == 0) {
        // Reset queue
        *tail_bin = 0;
        *count = 0;
        reached = 0;
    }

    // Fetch frontier elements from the queue
    if(tid == 0)
        *base = atomicAdd(&head[0], WG_SIZE);
    __syncthreads();

    int my_base = *base;
    while(my_base < n_t_local) {
        if(my_base + tid < n_t_local && *overflow == 0) {
            // Visit a node from the current frontier
            int pid = q1[my_base + tid];
            //////////////// Visit node ///////////////////////////
            atomicExch(&color[pid], BLACK); // Node visited
            int  cur_cost = atomicAdd(&cost[pid], 0); // Look up shortest-path distance to this node
            Node cur_node;
            cur_node.x = graph_nodes_av[pid].x;
            cur_node.y = graph_nodes_av[pid].y;
            Edge cur_edge;
            // For each outgoing edge
            for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                cur_edge.x = graph_edges_av[i].x;
                cur_edge.y = graph_edges_av[i].y;
                int id     = cur_edge.x;
                int cost_local   = cur_edge.y;
                cost_local += cur_cost;
                int orig_cost = atomicMax(&cost[id], cost_local);
                if(orig_cost < cost_local) {
                    int old_color = atomicMax(&color[id], gray_shade_local);
                    if(old_color != gray_shade_local) {
                        // Push to the queue
                        int tail_index = atomicAdd(tail_bin, 1);
                        while(tail_index >= W_QUEUE_SIZE) {
		                	*count = 1;
		                	shift_to_global(shift, tail, tail_bin, q2, l_q2, count, &reached);
		                    //*overflow = 1;
		                    tail_index = atomicAdd(tail_bin, 1);
                        }
                        l_q2[tail_index] = id;
                    }
                }
            }
        }
		while(shift_to_global(shift, tail, tail_bin, q2, l_q2, count, &reached));
        if(tid == 0)
            *base = atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
        __syncthreads();
        my_base = *base;
    }
    
	*count = 1;
    while(shift_to_global(shift, tail, tail_bin, q2, l_q2, count, &reached));

    if(gtid == 0) {
        atomicAdd(&iter[0], 1);
    }
}

cudaError_t call_SSSP_gpu(int blocks, int threads, Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *gray_shade, int *iter, int LIMIT, const int CPU, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    SSSP_gpu<<<dimGrid, dimBlock, l_mem_size>>>(graph_nodes_av, graph_edges_av, cost,
        color, q1, q2, n_t,
        head, tail, threads_end, threads_run,
        overflow, gray_shade, iter, LIMIT, CPU);
    
    cudaError_t err = cudaGetLastError();
    return err;
}
