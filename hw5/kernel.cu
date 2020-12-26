#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int thread_num=800;

__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    float new_im;
    int i;
    float zresq, zimsq;
    for (i = 0; i < count; ++i)
    {
        zresq = z_re*z_re;
        zimsq = z_im*z_im;
        if (zresq + zimsq > 4.f)
            break;

        new_im = z_re * z_im;
        z_re = c_re + zresq - zimsq;
        z_im = c_im + new_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *cu_img, int width, int maxIterations, int thread_num, size_t pitch_img) {
    int x_idx, y_idx, tot;
    tot = blockIdx.x*thread_num+threadIdx.x;
    x_idx = tot%(width);
    y_idx = tot/width;
    
    // x_idx = threadIdx.x+((blockIdx.x&0x1)*800);
    // y_idx = blockIdx.x>>1;

    *((int*)((char*)cu_img+(y_idx*pitch_img))+x_idx) = mandel(lowerX + x_idx * stepX, lowerY + y_idx * stepY, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *cu_img;
    size_t pitch_img;
    cudaMallocPitch((void**)&cu_img, &pitch_img, sizeof(int)*resX, resY);
    // printf("%d %d\n", resX, resY);
    mandelKernel<<<resX*resY/thread_num, thread_num>>>(lowerX, lowerY, stepX, stepY, cu_img, resX, maxIterations, thread_num, pitch_img);

    cudaMemcpy2D(img, resX*sizeof(int), cu_img, pitch_img, sizeof(int)*resX, resY, cudaMemcpyDeviceToHost);
}