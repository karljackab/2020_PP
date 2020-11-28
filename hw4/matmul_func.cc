#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mpi.h>

void print_mat(const int *a_mat, const int n, const int m, int fd=1){
    int mm1 = m-1;
    for(int i=0; i<n; i++){
        for(int j=0; j<mm1; j++)
            dprintf(fd, "%d ", a_mat[i*m+j]);
        dprintf(fd, "%d\n", a_mat[i*m+mm1]);
    }
}

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int a_len, b_len;
    if(world_rank){
        MPI_Recv(n_ptr, 1, MPI_INT, world_rank-1, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(m_ptr, 1, MPI_INT, world_rank-1, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(l_ptr, 1, MPI_INT, world_rank-1, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        a_len = *n_ptr * *m_ptr;
        *a_mat_ptr = (int*)malloc(sizeof(int*) * a_len);
        b_len = *m_ptr * *l_ptr;
        *b_mat_ptr = (int*)malloc(sizeof(int*) * b_len);
        MPI_Recv(*a_mat_ptr, a_len, MPI_INT, world_rank-1, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(*b_mat_ptr, b_len, MPI_INT, world_rank-1, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        a_len = *n_ptr * *m_ptr;
        *a_mat_ptr = (int*)malloc(sizeof(int*) * a_len);
        for(int i=0; i<a_len; i++)
            scanf("%d", &(*a_mat_ptr)[i]);

        b_len = *m_ptr * *l_ptr;
        *b_mat_ptr = (int*)malloc(sizeof(int*) * b_len);
        for(int i=0; i<b_len; i++)
            scanf("%d", &(*b_mat_ptr)[i]);
    }
    int target = world_rank+1;
    if(target<world_size){
        MPI_Send(n_ptr, 1, MPI_INT, target, target, MPI_COMM_WORLD);
        MPI_Send(m_ptr, 1, MPI_INT, target, target, MPI_COMM_WORLD);
        MPI_Send(l_ptr, 1, MPI_INT, target, target, MPI_COMM_WORLD);
        MPI_Send(*a_mat_ptr, a_len, MPI_INT, target, target, MPI_COMM_WORLD);
        MPI_Send(*b_mat_ptr, b_len, MPI_INT, target, target, MPI_COMM_WORLD);
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    setvbuf(stdout, NULL, _IOFBF, 2147483647);

    int world_rank, world_size, sub_n, main_n;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_size > 2)
        sub_n = n/(world_size-1);
    else
        sub_n = n/world_size;

    main_n = n-sub_n*(world_size-1);
    
    if(world_rank){
        int *res_mat = (int*)malloc(sizeof(int*)*sub_n*l);
        memset(res_mat, 0, sizeof(int*)*sub_n*l);
        int shift = main_n + (world_rank-1)*sub_n;
        for(int i=0, new_i=shift; i<sub_n; i++, new_i++)
            for(int j=0; j<l; j++){
                int sum=0, im=new_i*m;
                for(int k=0; k<m; k++)
                    sum += a_mat[im+k]*b_mat[k*l+j];
                res_mat[i*l+j] = sum;
            }

        MPI_Send((void*)res_mat, sub_n*l, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
    }
    else{
        int *res_mat = (int*)malloc(sizeof(int*)*n*l);
        memset(res_mat, 0, sizeof(int*)*n*l);
        for(int i=0; i<main_n; i++)
            for(int j=0; j<l; j++){
                int sum=0, im=i*m;
                for(int k=0; k<m; k++)
                    sum += a_mat[im+k]*b_mat[k*l+j];
                res_mat[i*l+j] = sum;
            }

        print_mat(res_mat, main_n, l);

        for(int source=1; source<world_size; source++){
            MPI_Recv(&res_mat[main_n*l+(source-1)*sub_n*l], sub_n*l, MPI_INT, source, source, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_mat(&res_mat[main_n*l+(source-1)*sub_n*l], sub_n, l);
        }
        fflush(stdout);
    }
}

void destruct_matrices(int *a_mat, int *b_mat){
    return;
}