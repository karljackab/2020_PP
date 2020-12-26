#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int thread_num=800;

__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *cu_img, int width, int maxIterations, int thread_num, size_t pitch_img) {
    int x_idx, y_idx, tot;
    tot = blockIdx.x*thread_num+threadIdx.x;
    x_idx = tot%(width);
    y_idx = tot/(width);

    float x = lowerX + x_idx * stepX;
    float y = lowerY + y_idx * stepY;

    int cur_res = mandel(x, y, maxIterations);
    *((int*)((char*)cu_img+(y_idx*pitch_img))+x_idx) = cur_res;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int tot_len = resX*resY;

    int *value;
    cudaHostAlloc((void**)&value, tot_len*sizeof(int), cudaHostAllocDefault);

    int *cu_img;
    size_t pitch_img;
    cudaMallocPitch((void**)&cu_img, &pitch_img, sizeof(int)*resX, resY);
    
    mandelKernel<<<tot_len/thread_num, thread_num>>>(lowerX, lowerY, stepX, stepY, cu_img, resX, maxIterations, thread_num, pitch_img);

    cudaMemcpy2D(value, resX*sizeof(int), cu_img, pitch_img, sizeof(int)*resX, resY, cudaMemcpyDeviceToHost);
    memcpy(img, value, sizeof(int)*tot_len);
}