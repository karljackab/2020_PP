#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <unordered_map>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int *cnt_ptr = &new_frontier->count;
                int index = __sync_fetch_and_add(cnt_ptr, 1);
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

bool bottom_up_step(
    Graph g,
    bool *bit_frontier,
    bool *new_bit_frontier,
    int *distances)
{
    bool still_remain = false;
    #pragma omp parallel for
    for (int node = 0; node < g->num_nodes; node++)
    {
        if (distances[node] != NOT_VISITED_MARKER)
            continue;

        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)? g->num_edges: g->incoming_starts[node + 1];

        for(int neighbor=start_edge; neighbor < end_edge; neighbor++){
            if(bit_frontier[g->incoming_edges[neighbor]]){
                distances[node] = distances[g->incoming_edges[neighbor]] + 1;
                new_bit_frontier[node] = true;
                if(!still_remain)
                    still_remain = true;
                break;
            }
        }
    }

    return still_remain;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    bool *bit_frontier, *new_bit_frontier;
    bit_frontier = (bool*)malloc(sizeof(bool)*graph->num_nodes);
    new_bit_frontier = (bool*)malloc(sizeof(bool)*graph->num_nodes);
    memset(bit_frontier, 0, sizeof(bool)*graph->num_nodes);
    memset(new_bit_frontier, 0, sizeof(bool)*graph->num_nodes);

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    bool still_remain = true;
    sol->distances[ROOT_NODE_ID] = 0;
    bit_frontier[ROOT_NODE_ID] = true;
    while(still_remain)
    {
        still_remain = bottom_up_step(graph, bit_frontier, new_bit_frontier, sol->distances);
        bool *tmp = bit_frontier;
        bit_frontier = new_bit_frontier;
        new_bit_frontier = tmp;
    }

}

void bfs_hybrid(Graph graph, solution *sol)
{
    int visited_node_cnt = 0, beta=6;

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *bit_frontier, *new_bit_frontier;
    bit_frontier = (bool*)malloc(sizeof(bool)*graph->num_nodes);
    new_bit_frontier = (bool*)malloc(sizeof(bool)*graph->num_nodes);
    memset(bit_frontier, 0, sizeof(bool)*graph->num_nodes);
    memset(new_bit_frontier, 0, sizeof(bool)*graph->num_nodes);

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    bit_frontier[ROOT_NODE_ID] = true;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    while(visited_node_cnt <= graph->num_nodes/beta){
        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);
        
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        #pragma omp parallel for
        for(int i=0; i<frontier->count; i++)
            bit_frontier[frontier->vertices[i]] = true;
        visited_node_cnt += frontier->count;
    }

    // printf("Switch when visited_node_cnt ratio %d/%d\n", visited_node_cnt, graph->num_nodes);

    bool still_remain = true;
    while(still_remain)
    {
        still_remain = bottom_up_step(graph, bit_frontier, new_bit_frontier, sol->distances);
        bool *tmp = bit_frontier;
        bit_frontier = new_bit_frontier;
        new_bit_frontier = tmp;
    }
}