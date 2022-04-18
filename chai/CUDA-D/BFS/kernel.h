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

#include "cuda_runtime.h"
#include <atomic>
#include "support/common.h"

void run_cpu_threads(Node *graph_nodes_av, Edge *graph_edges_av, std::atomic_llong *cost, std::atomic_llong *color,
    ll *q1, ll *q2, ll *t, std::atomic_llong *head, std::atomic_llong *tail,
    std::atomic_llong *threads_end, std::atomic_llong *threads_run, std::atomic_llong *iter, ll cpu_threads,
    ll LIMIT, const ll GPU);

cudaError_t call_BFS_gpu(ll blocks, ll threads, Node *graph_nodes_av, Edge *graph_edges_av, ll *cost,
    ll *color, ll *q1, ll *q2, ll *n_t,
    ll *head, ll *tail, ll *threads_end, ll *threads_run,
		ll *overflow, ll *iter, ll LIMIT, const ll CPU, ll l_mem_size);
