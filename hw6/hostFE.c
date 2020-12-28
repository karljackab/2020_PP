#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "hostFE.h"
#include "helper.h"

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void error_exit(char *msg, cl_int err) {
    printf("ERROR: %s - %s\n", msg, getErrorString(err));
    // std::cerr << "[Error] " << msg << " : " << err << "\n"; 
    exit(1);
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    FILE *f = fopen("./kernel.cl", "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *source = (char*)malloc(fsize + 1);
    fread(source, 1, fsize, f);
    fclose(f);
    source[fsize] = 0;

    size_t source_len = strlen(source);

    cl_platform_id platform_id;
    cl_command_queue command_queue;
    cl_mem args_inputImage, args_outputImage, args_filter;
    // cl_mem args_img_data, args_R, args_G, args_B;
    cl_kernel kern;
    cl_int err_code;

    size_t global_work_size = 2400; // :)
    size_t local_work_size = 100;    // ??
    unsigned int task_num = imageSize/global_work_size + (imageSize % global_work_size != 0);

    clGetPlatformIDs(1, &platform_id, NULL);
    // if (err_code != CL_SUCCESS) error_exit("clGetPlatformIDs()", err_code);

    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, device, NULL);
    // if (err_code != CL_SUCCESS) error_exit("clGetDeviceIDs()", err_code);

    context = clCreateContext(NULL, 1, device, NULL, NULL, &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateContext()", err_code);

    command_queue = clCreateCommandQueueWithProperties(context, *device, 0, &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateCommandQueueWithProperties()", err_code);

    args_inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*imageSize, NULL, &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);
    args_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*filterSize, NULL, &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);
    args_outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*imageSize, NULL, &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);

    clEnqueueWriteBuffer(command_queue, args_inputImage, CL_TRUE, 0, sizeof(float)*imageSize, inputImage, 0, NULL, NULL);
    // if (err_code != CL_SUCCESS) error_exit("clEnqueueWriteBuffer()", err_code);
    clEnqueueWriteBuffer(command_queue, args_filter, CL_TRUE, 0, sizeof(float)*filterSize, filter, 0, NULL, NULL);
    // if (err_code != CL_SUCCESS) error_exit("clEnqueueWriteBuffer()", err_code);

    *program = clCreateProgramWithSource(context, 1, &source, &source_len, &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateProgramWithSource()", err_code);

    clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
    // if (err_code != CL_SUCCESS) {
    //     size_t len = 0;
    //     cl_int err_build;
    //     err_build = clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    //     char *buffer = (char*)malloc(sizeof(char)*len);
    //     err_build = clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    //     dprintf(2, "%s", buffer);
    //     free(buffer);
    //     error_exit("clBuildProgram()", err_code);
    // }
    kern = clCreateKernel(*program, "convolution", &err_code);
    // if (err_code != CL_SUCCESS) error_exit("clCreateKernel()", err_code);

    // Pass Arguments
    clSetKernelArg(kern, 0, sizeof(cl_mem), &args_inputImage);
    // if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &args_outputImage);
    // if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
    clSetKernelArg(kern, 2, sizeof(cl_mem), &args_filter);
    // if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
    clSetKernelArg(kern, 3, sizeof(int), &filterWidth);
    // if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
    clSetKernelArg(kern, 4, sizeof(int), &imageHeight);
    // if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
    clSetKernelArg(kern, 5, sizeof(int), &imageWidth);
    // if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);

    clEnqueueNDRangeKernel(command_queue, kern, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    // if (err_code != CL_SUCCESS) error_exit("clEnqueueNDRangeKernel()", err_code);

    // read data from device to host
    clEnqueueReadBuffer(command_queue, args_outputImage, CL_TRUE, 0, sizeof(float)*imageSize, outputImage, 0, NULL, NULL);
    // if (err_code != CL_SUCCESS) error_exit("clEnqueueReadBuffer()", err_code);

    // release
    // clReleaseProgram(program);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseProgram()", err_code);
    // clReleaseKernel(kern);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseKernel()", err_code);
    // clReleaseCommandQueue(command_queue);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseCommandQueue()", err_code);
    // clReleaseContext(context);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseContext()", err_code);
    // clReleaseMemObject(args_inputImage);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);
    // clReleaseMemObject(args_filter);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);
    // clReleaseMemObject(args_outputImage);
    // if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);
}