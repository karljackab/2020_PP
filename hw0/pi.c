#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    long long int number_of_tosses = 1000000000;
    long long int number_in_circle = 0;
    srand(time(NULL));
    long long int NEW_RAND_MAX = (long long int)RAND_MAX << 32 | (long long int)RAND_MAX;
    for(int toss=0; toss<number_of_tosses; toss++)
    {
        double x = (double)((long long int)rand() << 32 | (long long int)rand()) / NEW_RAND_MAX;
        double y = (double)((long long int)rand() << 32 | (long long int)rand()) / NEW_RAND_MAX;
        if((x*x + y*y) <= 1)
            number_in_circle += 1;
    }
    double pi_esti = 4*number_in_circle / ((double)number_of_tosses);
    printf("%.8lf\n", pi_esti);
    return 0;
}