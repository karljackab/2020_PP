#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

typedef unsigned long long int ull;

ull child_toss(ull toss_cnt, int world_rank){
    ull number_in_circle=0;
    double x, y;
    unsigned seed = (unsigned)time(NULL)*world_rank;
    ull new_rand_max = (ull)RAND_MAX*RAND_MAX;
    for(ull toss=0; toss<toss_cnt; toss++){
        x = (double)rand_r(&seed);
        y = (double)rand_r(&seed);
        if((x*x + y*y) <= new_rand_max)
            number_in_circle += 1;
    }
    return number_in_circle;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    ull in_circle;
    MPI_Request *request;
    ull *rec_in_circle_list;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    request = (MPI_Request*)malloc(sizeof(MPI_Request)*world_size);
    rec_in_circle_list = (ull*)malloc(sizeof(ull)*world_size);
    // TODO: init MPI

    if (world_rank > 0)
    {
        in_circle = child_toss(tosses/world_size, world_rank);
        MPI_Send((void*)&in_circle, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        for(int source=1; source<world_size; source++)
            MPI_Irecv(&rec_in_circle_list[source], 1, MPI_UNSIGNED_LONG, source, 0, MPI_COMM_WORLD, &request[source]);
        in_circle = child_toss(tosses/world_size, world_rank);
    }
    // char processor_name[MPI_MAX_PROCESSOR_NAME];
    // int name_len;
    // MPI_Get_processor_name(processor_name, &name_len);
    // printf("Process %d at %s: in_circle %llu\n", world_rank, processor_name, in_circle);
    
    if (world_rank == 0)
    {
        ull tot_toss = tosses;
        MPI_Waitall(world_size-1, &request[1], MPI_STATUSES_IGNORE);
        for(int source=1; source<world_size; source++)
            in_circle += rec_in_circle_list[source];
        pi_result = 4*in_circle / ((double)tot_toss);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
