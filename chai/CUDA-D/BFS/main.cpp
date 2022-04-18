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

#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    ll         device;
    ll         n_gpu_threads;
    ll         n_gpu_blocks;
    ll         n_threads;
    ll         n_warmup;
    ll         n_reps;
    const char *file_name;
    const char *comparison_file;
    ll         switching_limit;

    Params(int argc, char **argv) {
        device          = 0;
        n_gpu_threads    = 512;
        n_gpu_blocks   = 320;
        n_threads       = 0;
        n_warmup        = 0;
        n_reps          = 1;
        file_name       = "input/NYR_input.dat";
        comparison_file = "output/NYR_bfs_BFS.out";
        switching_limit = 1;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_gpu_threads    = atoi(optarg); break;
            case 'g': n_gpu_blocks   = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            case 'l': switching_limit = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        //assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        //assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        //assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./bfs [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=8)"
                "\n    -t <T>    # of host threads (default=2)"
                "\n    -w <W>    # of untimed warmup iterations (default=1)"
                "\n    -r <R>    # of timed repetition iterations (default=1)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    name of input file with control points (default=input/NYR_input.dat)"
                "\n    -c <C>    comparison file (default=output/NYR_bfs_BFS.out)"
                "\n    -l <L>    switching limit (default=128)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input_size(ll &n_nodes, ll &n_edges, const Params &p) {
    FILE *fp = fopen(p.file_name, "r");
    fscanf(fp, "%lld", &n_nodes);
    fscanf(fp, "%lld", &n_edges);
    if(fp)
        fclose(fp);
}

void read_input(ll &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

    ll   start, edgeno;
    ll   n_nodes, n_edges;
    ll   id, cost;
    FILE *fp = fopen(p.file_name, "r");

    fscanf(fp, "%lld", &n_nodes);
    fscanf(fp, "%lld", &n_edges);
    fscanf(fp, "%lld", &source);
    printf("Number of nodes = %lld\t", n_nodes);
    printf("Number of edges = %lld\t", n_edges);

    // initalize the memory: Nodes
    for(ll i = 0; i < n_nodes; i++) {
        fscanf(fp, "%lld %lld", &start, &edgeno);
        h_nodes[i].x = start;
        h_nodes[i].y = edgeno;
    }
#if PRINT_ALL
    for(ll i = 0; i < n_nodes; i++) {
        printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y);
    }
#endif

    // initalize the memory: Edges
    for(ll i = 0; i < n_edges; i++) {
        fscanf(fp, "%lld", &id);
        fscanf(fp, "%lld", &cost);
        h_edges[i].x = id;
        h_edges[i].y = -cost;
    }
    if(fp)
        fclose(fp);
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    CUDASetup    setcuda(p.device);
    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    ll n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);
    timer.start("Allocation");
    Node * h_nodes = (Node *)malloc(sizeof(Node) * n_nodes);
		Node * d_nodes;
    cudaStatus = cudaMalloc((void**)&d_nodes, sizeof(Node) * n_nodes);
    Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);
		Edge * d_edges;
    cudaStatus = cudaMalloc((void**)&d_edges, sizeof(Edge) * n_edges);
    std::atomic_llong *h_color = (std::atomic_llong *)malloc(sizeof(std::atomic_llong) * n_nodes);
		ll * d_color;
    cudaStatus = cudaMalloc((void**)&d_color, sizeof(ll) * n_nodes);
    std::atomic_llong *h_cost  = (std::atomic_llong *)malloc(sizeof(std::atomic_llong) * n_nodes);
		ll * d_cost;
    cudaStatus = cudaMalloc((void**)&d_cost, sizeof(ll) * n_nodes);
    ll *            h_q1    = (ll *)malloc(n_nodes * sizeof(ll));
		ll * d_q1;
    cudaStatus = cudaMalloc((void**)&d_q1, sizeof(ll) * n_nodes);
    ll *            h_q2    = (ll *)malloc(n_nodes * sizeof(ll));
		ll * d_q2;
    cudaStatus = cudaMalloc((void**)&d_q2, sizeof(ll) * n_nodes);
    std::atomic_llong  h_head[1];
		ll * d_head;
    cudaStatus = cudaMalloc((void**)&d_head, sizeof(ll));
    std::atomic_llong  h_tail[1];
		ll * d_tail;
    cudaStatus = cudaMalloc((void**)&d_tail, sizeof(ll));
    std::atomic_llong  h_threads_end[1];
		ll * d_threads_end;
    cudaStatus = cudaMalloc((void**)&d_threads_end, sizeof(ll));
    std::atomic_llong  h_threads_run[1];
		ll * d_threads_run;
    cudaStatus = cudaMalloc((void**)&d_threads_run, sizeof(ll));
    ll              h_num_t[1];
		ll * d_num_t;
    cudaStatus = cudaMalloc((void**)&d_num_t, sizeof(ll));
    ll              h_overflow[1];
		ll * d_overflow;
    cudaStatus = cudaMalloc((void**)&d_overflow, sizeof(ll));
    std::atomic_llong  h_iter[1];
		ll * d_iter;
    cudaStatus = cudaMalloc((void**)&d_iter, sizeof(ll));
    ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);
    CUDA_ERR();
    cudaDeviceSynchronize();
    timer.stop("Allocation");

    // Initialize
    timer.start("Initialization");
    const ll max_gpu_threads = setcuda.max_gpu_threads();
    ll source;
    read_input(source, h_nodes, h_edges, p);
    for(ll i = 0; i < n_nodes; i++) {
        h_cost[i].store(INF);
    }
    h_cost[source].store(0);
    for(ll i = 0; i < n_nodes; i++) {
        h_color[i].store(WHITE);
    }
    h_tail[0].store(0);
    h_head[0].store(0);
    h_threads_end[0].store(0);
    h_threads_run[0].store(0);
    h_q1[0] = source;
    h_iter[0].store(0);
    h_overflow[0] = 0;
    cudaDeviceSynchronize();
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    // Copy to device
    timer.start("Copy To Device");
    cudaStatus =
        cudaMemcpy(d_nodes, h_nodes, sizeof(Node) * n_nodes, cudaMemcpyHostToDevice);
    cudaStatus =
        cudaMemcpy(d_edges, h_edges, sizeof(Edge) * n_edges, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CUDA_ERR();
    timer.stop("Copy To Device");

    for(ll rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        for(ll i = 0; i < n_nodes; i++) {
            h_cost[i].store(INF);
        }
        h_cost[source].store(0);
        for(ll i = 0; i < n_nodes; i++) {
            h_color[i].store(WHITE);
        }
        h_tail[0].store(0);
        h_head[0].store(0);
        h_threads_end[0].store(0);
        h_threads_run[0].store(0);
        h_q1[0] = source;
        h_iter[0].store(0);
        h_overflow[0] = 0;
        h_num_t[0] = 1;
		/*
        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Run first iteration in master CPU thread
        ll pid;
        ll index_i, index_o;
        for(index_i = 0; index_i < h_num_t[0]; index_i++) {
            pid = h_q1[index_i];
            h_color[pid].store(BLACK);
            for(ll i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x); i++) {
                ll id = h_edges[i].x;
                h_color[id].store(BLACK);
                index_o       = h_tail[0].fetch_add(1);
                h_q2[index_o] = id;
            }
        }
        h_num_t[0] = h_tail[0].load();
        h_tail[0].store(0);
        h_threads_run[0].fetch_add(1);
        h_iter[0].fetch_add(1);
        if(rep >= p.n_warmup)
            timer.stop("Kernel");
		*/
        // Pointers to input and output queues
        ll * h_qin  = h_q2;
        ll * h_qout = h_q1;
        ll * d_qin  = d_q2;
        ll * d_qout = d_q1;

        const ll CPU_EXEC = (p.n_threads > 0) ? 1 : 0;
        const ll GPU_EXEC = (p.n_gpu_blocks > 0 && p.n_gpu_threads > 0) ? 1 : 0;

        // Run subsequent iterations on CPU or GPU until number of input queue elements is 0
        while(*h_num_t != 0) {

            if((*h_num_t < p.switching_limit || GPU_EXEC == 0) &&
                CPU_EXEC == 1) { // If the number of input queue elements is lower than switching_limit

                if(rep >= p.n_warmup)
                    timer.start("Kernel");

                // Continue until switching_limit condition is not satisfied
                while((*h_num_t != 0) && (*h_num_t < p.switching_limit || GPU_EXEC == 0) && CPU_EXEC == 1) {

                    // Swap queues
                    if(h_iter[0] % 2 == 0) {
                        h_qin  = h_q1;
                        h_qout = h_q2;
                    } else {
                        h_qin  = h_q2;
                        h_qout = h_q1;
                    }

                    std::thread main_thread(run_cpu_threads, h_nodes, h_edges, h_cost, h_color, h_qin, h_qout, h_num_t,
                        h_head, h_tail, h_threads_end, h_threads_run, h_iter, p.n_threads, p.switching_limit, GPU_EXEC);
                    main_thread.join();

                    h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
                    h_tail[0].store(0);
                    h_head[0].store(0);
                }

                if(rep >= p.n_warmup)
                    timer.stop("Kernel");

            } else if((*h_num_t >= p.switching_limit || CPU_EXEC == 0) &&
                      GPU_EXEC ==
                          1) { // If the number of input queue elements is higher than or equal to switching_limit

                if(rep >= p.n_warmup)
                    timer.start("Copy To Device");
                cudaStatus = cudaMemcpy(
                    d_cost, h_cost, sizeof(ll) * n_nodes, cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_color, h_color, sizeof(ll) * n_nodes, cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_threads_run, h_threads_run, sizeof(ll), cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_threads_end, h_threads_end, sizeof(ll), cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_overflow, h_overflow, sizeof(ll), cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_q1, h_q1, sizeof(ll) * n_nodes, cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_q2, h_q2, sizeof(ll) * n_nodes, cudaMemcpyHostToDevice);
                cudaStatus = cudaMemcpy(
                    d_iter, h_iter, sizeof(ll), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                CUDA_ERR();
                if(rep >= p.n_warmup)
                    timer.stop("Copy To Device");

				int iter = 0;
                // Continue until switching_limit condition is not satisfied
                while((*h_num_t != 0) && (*h_num_t >= p.switching_limit || CPU_EXEC == 0) && GPU_EXEC == 1) {

                    // Swap queues
                    if(h_iter[0] % 2 == 0) {
                        d_qin  = d_q1;
                        d_qout = d_q2;
                    } else {
                        d_qin  = d_q2;
                        d_qout = d_q1;
                    }

                    if(rep >= p.n_warmup)
                        timer.start("Copy To Device");
                    cudaStatus = cudaMemcpy(
                        d_num_t, h_num_t, sizeof(ll), cudaMemcpyHostToDevice);
                    cudaStatus = cudaMemcpy(
                        d_tail, h_tail, sizeof(ll), cudaMemcpyHostToDevice);
                    cudaStatus = cudaMemcpy(
                        d_head, h_head, sizeof(ll), cudaMemcpyHostToDevice);
                    cudaDeviceSynchronize();
                    CUDA_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Copy To Device");

                    if(rep >= p.n_warmup)
                        timer.start("Kernel");
                    assert(p.n_gpu_threads <= max_gpu_threads && 
                        "The thread block size is greater than the maximum thread block size that can be used on this device");
                    
                    ll threadblocks = min(p.n_gpu_blocks, (*h_num_t + p.n_gpu_threads - 1) / p.n_gpu_threads);
                    
                    cudaStatus = call_BFS_gpu(threadblocks, p.n_gpu_threads, d_nodes, d_edges, d_cost,
                        d_color, d_qin, d_qout, d_num_t,
                        d_head, d_tail, d_threads_end, d_threads_run,
                    		d_overflow, d_iter, p.switching_limit, CPU_EXEC, sizeof(ll) * (W_QUEUE_SIZE + 3));
                    cudaDeviceSynchronize();
                    CUDA_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Kernel");

                    if(rep >= p.n_warmup)
                        timer.start("Copy Back and Merge");
                    cudaStatus = cudaMemcpy(
                        h_tail, d_tail, sizeof(ll), cudaMemcpyDeviceToHost);
                    cudaStatus = cudaMemcpy(
                        h_iter, d_iter, sizeof(ll), cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    CUDA_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Copy Back and Merge");

                    h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
                    h_tail[0].store(0);
                    h_head[0].store(0);
                    printf("%d\tvisited\t%lld\n", ++iter, h_num_t[0]);
                }

                if(rep >= p.n_warmup)
                    timer.start("Copy Back and Merge");
                cudaStatus = cudaMemcpy(
                    h_cost, d_cost, sizeof(ll) * n_nodes, cudaMemcpyDeviceToHost);
                cudaStatus = cudaMemcpy(
                    h_color, d_color, sizeof(ll) * n_nodes, cudaMemcpyDeviceToHost);
                cudaStatus = cudaMemcpy(
                    h_threads_run, d_threads_run, sizeof(ll), cudaMemcpyDeviceToHost);
                cudaStatus = cudaMemcpy(
                    h_threads_end, d_threads_end, sizeof(ll), cudaMemcpyDeviceToHost);
                cudaStatus = cudaMemcpy(
                    h_overflow, d_overflow, sizeof(ll), cudaMemcpyDeviceToHost);
                cudaStatus = cudaMemcpy(
                    h_q1, d_q1, sizeof(ll) * n_nodes, cudaMemcpyDeviceToHost);
                cudaStatus = cudaMemcpy(
                    h_q2, d_q2, sizeof(ll) * n_nodes, cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                CUDA_ERR();
                if(rep >= p.n_warmup)
                    timer.stop("Copy Back and Merge");
            }
        }

    } // end of iteration
    timer.print("Allocation", 1);
    timer.print("Copy To Device", p.n_reps);
    timer.print("Kernel", p.n_reps);
    timer.print("Copy Back and Merge", p.n_reps);

    // Verify answer
    //verify(h_cost, n_nodes, p.comparison_file);

    // Free memory
    timer.start("Deallocation");
    free(h_nodes);
    free(h_edges);
    free(h_color);
    free(h_cost);
    free(h_q1);
    free(h_q2);
    cudaStatus = cudaFree(d_nodes);
    cudaStatus = cudaFree(d_edges);
    cudaStatus = cudaFree(d_cost);
    cudaStatus = cudaFree(d_color);
    cudaStatus = cudaFree(d_q1);
    cudaStatus = cudaFree(d_q2);
    cudaStatus = cudaFree(d_num_t);
    cudaStatus = cudaFree(d_head);
    cudaStatus = cudaFree(d_tail);
    cudaStatus = cudaFree(d_threads_end);
    cudaStatus = cudaFree(d_threads_run);
    cudaStatus = cudaFree(d_overflow);
    cudaStatus = cudaFree(d_iter);
    CUDA_ERR();
    cudaDeviceSynchronize();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    // Release timers
    timer.release("Allocation");
    timer.release("Initialization");
    timer.release("Copy To Device");
    timer.release("Kernel");
    timer.release("Copy Back and Merge");
    timer.release("Deallocation");

    printf("Test Passed\n");
    return 0;
}