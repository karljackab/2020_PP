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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // TODO: MPI init
    in_circle = child_toss(tosses/world_size, world_rank);

    // TODO: binary tree redunction
    int cur_rank=world_rank, merge_idx=2;
    ull rec_in_circle;
    MPI_Status status;
    int receive_cnt = 0, cal_rec_cnt=world_size-1;
    while(cal_rec_cnt){
        receive_cnt += 1;
        cal_rec_cnt >>= 1;
    }

    while(world_rank!=0 || receive_cnt>0){
        if(world_rank%merge_idx){
            MPI_Send((void*)&in_circle, 1, MPI_UNSIGNED_LONG, world_rank-(merge_idx>>1), merge_idx, MPI_COMM_WORLD);
            break;
        }
        else if(world_rank+(merge_idx>>1) < world_size){
            MPI_Recv(&rec_in_circle, 1, MPI_UNSIGNED_LONG, world_rank+(merge_idx>>1), merge_idx, MPI_COMM_WORLD, &status);
            in_circle += rec_in_circle;
            receive_cnt -= 1;
        }
        merge_idx <<= 1;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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
