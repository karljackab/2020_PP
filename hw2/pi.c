#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

typedef unsigned long long int ull;

void* child_toss(void* toss_cnt){
    ull number_in_circle=0, whole_toss_cnt=*(ull*)toss_cnt;
    unsigned seed = time(NULL);
    double x, y;
    for(ull toss=0; toss<whole_toss_cnt; toss++){
        x = (double)rand_r(&seed)/RAND_MAX;
        y = (double)rand_r(&seed)/RAND_MAX;
        if((x*x + y*y) <= 1)
            number_in_circle += 1;
    }
    return (void*)number_in_circle;
}

int main(int argc, char *argv[])
{
    int thread_num = atoi(argv[1]);
    ull toss_num = atoll(argv[2]);

    pthread_t *thread;
    thread = (pthread_t*)malloc(sizeof(pthread_t)*thread_num);

    int cur_thread_id;
    ull sub_toss_num=toss_num/thread_num;
    for(cur_thread_id=0; cur_thread_id<thread_num; cur_thread_id++)
        pthread_create(thread+cur_thread_id, NULL, child_toss, (void*)&sub_toss_num);

    ull number_in_circle = 0;
    void* sub_number;
    for(cur_thread_id=0; cur_thread_id<thread_num; cur_thread_id++){
        pthread_join(thread[cur_thread_id], &sub_number);
        number_in_circle += (long long int)sub_number;
    }
    printf("%.8lf\n", 4*number_in_circle / ((double)toss_num));
    return 0;
}