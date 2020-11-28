#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

unsigned long int child_toss(unsigned long int toss_cnt, int world_rank){
    unsigned long int number_in_circle=0;
    double x, y;
    unsigned seed = (unsigned)time(NULL)*world_rank;
    unsigned long int new_rand_max = (unsigned long int)RAND_MAX*RAND_MAX;
    for(unsigned long int toss=0; toss<toss_cnt; toss++){
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

    // TODO: MPI init
    unsigned long int in_circle, *rbuf;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    in_circle = child_toss(tosses/world_size, world_rank);
    // TODO: use MPI_Gather
    if(world_rank == 0)
        rbuf = (unsigned long int*)malloc(sizeof(unsigned long long int)*world_size);

    MPI_Gather(&in_circle, 1, MPI_UNSIGNED_LONG, rbuf, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); 

    if (world_rank == 0)
    {
        unsigned long int tot=0;
        for(int i=0; i<world_size; i++)
            tot += rbuf[i];
            
        // TODO: PI result
        pi_result = 4*tot / ((double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
