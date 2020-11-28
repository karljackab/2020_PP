#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

typedef unsigned long long int ul;

ul child_toss(ul toss_cnt, int world_rank){
    ul number_in_circle=0;
    double x, y;
    unsigned seed = (unsigned)time(NULL)*world_rank;
    ul new_rand_max = (ul)RAND_MAX*RAND_MAX;
    for(ul toss=0; toss<toss_cnt; toss++){
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

    MPI_Win win;

    // TODO: MPI init
    ul in_circle;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    in_circle = child_toss(tosses/world_size, world_rank);

    if (world_rank == 0)
    {
        ul *schedule;
        MPI_Alloc_mem(world_size * sizeof(ul), MPI_INFO_NULL, &schedule);
        
        for (int i = 0; i < world_size; i++)
            schedule[i] = 0;
        
        MPI_Win_create(schedule, world_size * sizeof(ul), sizeof(ul), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        int cnt=1;
        while(cnt!=world_size){
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            if(schedule[cnt]!=0){
                in_circle += schedule[cnt];
                cnt += 1;
            }
            MPI_Win_unlock(0, win);
        }

        MPI_Free_mem(schedule);
    }
    else
    {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&in_circle, 1, MPI_UNSIGNED_LONG, 0, world_rank, 1, MPI_UNSIGNED_LONG, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4*in_circle / ((double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}