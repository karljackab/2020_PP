#include <stdio.h>
#include <thread>

#include "CycleTimer.h"

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

///////////////////////////////////////////////////////////////////////////
static inline int mandel(float c_re, float c_im, int count)
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
void mandelbrotChange(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[], int step_size)
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  int endRow = startRow + totalRows;

  for (int j = startRow; j < endRow; j+=step_size)
  {
    for (int i = 0; i < width; ++i)
    {
      float x = x0 + i * dx;
      float y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
    }
  }
}
///////////////////////////////////////////////////////////////////////////


//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{
    // double startTime = CycleTimer::currentSeconds();

    // int cur_tot_row = args->height/args->numThreads;
    // int start_row_idx = cur_tot_row*args->threadId;
    // if(args->threadId == args->numThreads-1)
    //     cur_tot_row = args->height - cur_tot_row*args->threadId;
    // mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, 
    //     start_row_idx, cur_tot_row, args->maxIterations, args->output);

    // double endTime = CycleTimer::currentSeconds();
    // printf("Thread %d cost %f\n", args->threadId, endTime - startTime);


    mandelbrotChange(args->x0, args->y0, args->x1, args->y1, args->width, args->height, 
        args->threadId, args->height-args->threadId, args->maxIterations, args->output, args->numThreads);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
