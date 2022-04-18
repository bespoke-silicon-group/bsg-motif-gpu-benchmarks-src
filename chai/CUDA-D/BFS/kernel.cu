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
#define ull unsigned long long
#include <stdio.h>
#include <assert.h>

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void BFS_gpu(Node *graph_nodes_av, Edge *graph_edges_av, ll *cost,
    ll *color, ll *q1, ll *q2, ll *n_t,
    ll *head, ll *tail, ll *threads_end, ll *threads_run,
    ll *overflow, ll *iter, ll LIMIT, const ll CPU) {

    extern __shared__ ll l_mem[];
    ll* tail_bin = l_mem;
    ll* l_q2 = (ll*)&tail_bin[1];
    ll* shift = (ll*)&l_q2[W_QUEUE_SIZE];
    ll* base = (ll*)&shift[1];

    const ll tid     = threadIdx.x;
    const ll gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const ll MAXWG   = gridDim.x;
    const ll WG_SIZE = blockDim.x;

    ll iter_local = atomicAdd((ull*)&iter[0], 0);

    ll n_t_local = atomicAdd((ull*)n_t, 0);

    if(tid == 0) {
        // Reset queue
        *tail_bin = 0;
    }

    // Fetch frontier elements from the queue
    if(tid == 0) {
        *base = atomicAdd((ull*)&head[0], WG_SIZE);
    }
    __syncthreads();

    ll my_base = *base;
    while(my_base < n_t_local) {
        if(my_base + tid < n_t_local && *overflow == 0) {
            // Visit a node from the current frontier
            ll pid = q1[my_base + tid];
            //////////////// Visit node ///////////////////////////
            atomicExch((ull*)&cost[pid], iter_local); // Node visited
            Node cur_node;
            cur_node.x = graph_nodes_av[pid].x;
            cur_node.y = graph_nodes_av[pid].y;
            // For each outgoing edge
            for(ll i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                ll id        = graph_edges_av[i].x;
                ll old_color = atomicMax((ull*)&color[id], BLACK);
                if(old_color < BLACK) {
                    // Push to the queue
                    ll tail_index = atomicAdd((ull*)tail_bin, 1);
                    if(tail_index >= W_QUEUE_SIZE) {
                        *overflow = 1;
                        //printf("OVERFLOW\n");
                        assert(false);
                    } else
                        l_q2[tail_index] = id;
                }
            }
        }
        if(tid == 0) {
            *base = atomicAdd((ull*)&head[0], WG_SIZE); // Fetch more frontier elements from the queue
        }
        __syncthreads();
        my_base = *base;
    }
    /////////////////////////////////////////////////////////
    // Compute size of the output and allocate space in the global queue
    if(tid == 0) {
        *shift = atomicAdd((ull*)&tail[0], *tail_bin);
    }
    __syncthreads();
    ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
    ll local_shift = tid;
    while(local_shift < *tail_bin) {
        q2[*shift + local_shift] = l_q2[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += WG_SIZE;
    }
    //////////////////////////////////////////////////////////////////////////

    if(gtid == 0) {
        atomicAdd((ull*)&iter[0], 1);
    }
}

cudaError_t call_BFS_gpu(ll blocks, ll threads, Node *graph_nodes_av, Edge *graph_edges_av, ll *cost,
    ll *color, ll *q1, ll *q2, ll *n_t,
    ll *head, ll *tail, ll *threads_end, ll *threads_run,
    ll *overflow, ll *iter, ll LIMIT, const ll CPU, ll l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    BFS_gpu<<<dimGrid, dimBlock, l_mem_size>>>(graph_nodes_av, graph_edges_av, cost,
        color, q1, q2, n_t,
        head, tail, threads_end, threads_run,
        overflow, iter, LIMIT, CPU);
    
    cudaError_t err = cudaGetLastError();
    return err;
}
