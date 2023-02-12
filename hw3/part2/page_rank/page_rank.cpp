#include <iostream>
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
  //double equal_prob = 1.0 / numNodes;
  double *score_old = (double *)aligned_alloc(64, numNodes * sizeof(double));
	double *score_new = (double *)aligned_alloc(64, numNodes * sizeof(double));
  int num_no_outgoing = 0;
  int *no_outgoing_v = (int *)aligned_alloc(64, numNodes * sizeof(int));
	bool converged = false;
	double *temp = (double *)aligned_alloc(64, numNodes * sizeof(double));

  for (int i = 0; i < numNodes; i++)
  {
    solution[i] = 1.0 / (double)numNodes;
    score_old[i] = 1/(double)numNodes;
    score_new[i] = 0;
    temp[i] = score_old[i] / (double)outgoing_size(g,i); 

    if(outgoing_size(g,i) == 0){
		  no_outgoing_v[num_no_outgoing] = i;
			num_no_outgoing++;
		}
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

  while(!converged) {

    double no_outgoing_sum = 0.f;
    for(int i = 0; i < num_no_outgoing; i++)
      no_outgoing_sum += (damping * score_old[no_outgoing_v[i]] / (double)numNodes);
    

    #pragma omp parallel for
    for(int i = 0; i < numNodes; i++){

      for(int j = 0; j < incoming_size(g, i); j++){
        int vj = *(incoming_begin(g,i) + j);
        score_new[i] += temp[vj]; 
      }

      score_new[i] = (damping * score_new[i]) + (1.0-damping) / (double)numNodes;
      score_new[i] += no_outgoing_sum;
    }

    double global_diff = 0;
    for(int i = 0; i < numNodes; i++){
      global_diff += fabs(score_new[i] - score_old[i]);
      score_old[i] = score_new[i];
      temp[i] = score_old[i] / (double)outgoing_size(g, i);
      score_new[i] = 0;
    }

    converged = (global_diff < convergence);
  }

  //update solution
	#pragma omp parallel for
	for(int i = 0; i < numNodes; i++)
		solution[i] = score_old[i];

	free(score_old);
	free(score_new);
	free(no_outgoing_v);
	free(temp);

}
