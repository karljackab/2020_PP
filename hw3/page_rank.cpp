#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

	int numNodes = num_nodes(g);
	double *tmp_new_sol = (double*)malloc(sizeof(double)*numNodes);
	Vertex *empty_out_nodes = (Vertex*)malloc(sizeof(Vertex)*numNodes);
	int empty_len = 0;
	double equal_prob = 1.0 / numNodes;

	for (int i = 0; i < numNodes; ++i){
		solution[i] = equal_prob;
		tmp_new_sol[i] = 0;

		int out_begin = g->outgoing_starts[i];
		int out_end = (i == g->num_nodes - 1)? g->num_edges: g->outgoing_starts[i + 1];

		if(out_begin == out_end){
			empty_out_nodes[empty_len] = i;
			empty_len += 1;
		}
	}

	double global_diff = convergence+1;

	double partial_add = (1.0-damping)/numNodes;
	double no_out_nodes_val;
	int cnt = 0, start_edge, end_edge, incoming, out_begin, out_end;
	while(global_diff >= convergence){
		cnt += 1;
		no_out_nodes_val = 0.;
		#pragma omp parallel for reduction (+:no_out_nodes_val)
		for(int i=0; i<empty_len; i++){
			no_out_nodes_val += damping*solution[empty_out_nodes[i]]/numNodes;
		}

		global_diff = 0;

		#pragma omp parallel for reduction (+:global_diff) 
		for(int node=0; node < numNodes; node++){
			start_edge = g->incoming_starts[node];
			end_edge = (node == g->num_nodes - 1)? g->num_edges: g->incoming_starts[node + 1];

			// propogate information
			double sub_sum = 0.;
			for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
				incoming = g->incoming_edges[neighbor];
				out_begin = g->outgoing_starts[incoming];
				out_end = (incoming == g->num_nodes - 1)? g->num_edges: g->outgoing_starts[incoming + 1];
				sub_sum += solution[incoming]/(out_end-out_begin);
			}

			// add dumping & add empty outgoing node's value

			tmp_new_sol[node] = damping*sub_sum + partial_add + no_out_nodes_val;
			global_diff += abs(tmp_new_sol[node]-solution[node]);
		}
		// calculation global_diff and update new score
		#pragma omp parallel for
		for(int node=0; node<numNodes; node++)
			solution[node] = tmp_new_sol[node];
		memset(tmp_new_sol, 0, sizeof(double)*numNodes);
	}
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
