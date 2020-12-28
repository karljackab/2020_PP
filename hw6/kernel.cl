__kernel void convolution(__global float *inputImage, 
                        __global float *outputImage, 
                        __global float *filter, 
                        int filterWidth, 
                        int imageHeight, 
                        int imageWidth) {
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum;
    int i = 0, j = global_id, k, l;

    while(i < imageHeight){
        while(j >= imageWidth){
            j -= imageWidth;
            i += 1;
        }
        if(i >= imageHeight)
            break;

        sum = 0;
        // Apply the filter to the neighborhood
        for (k = -halffilterSize; k <= halffilterSize; k++)
        {
            for (l = -halffilterSize; l <= halffilterSize; l++)
            {
                if (i + k >= 0 && i + k < imageHeight &&
                    j + l >= 0 && j + l < imageWidth)
                {
                    sum += inputImage[(i + k) * imageWidth + j + l] *
                            filter[(k + halffilterSize) * filterWidth +
                                    l + halffilterSize];
                }
            }
        }
        outputImage[i * imageWidth + j] = sum;

        j += global_size;
    }
}
