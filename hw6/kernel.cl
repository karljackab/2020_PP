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

    if(j >= imageWidth){
        i += j/imageWidth;
        j %= imageWidth;
    }

    int first_term, img_first_term;
    while(i < imageHeight){
        sum = 0;

        // Apply the filter to the neighborhood
        first_term = 0;
        img_first_term = (i-halffilterSize)*imageWidth;
        for (k = -halffilterSize; k <= halffilterSize; k++, first_term+=filterWidth, img_first_term+=imageWidth)
            for (l = -halffilterSize; l <= halffilterSize; l++)
                if ((i + k >= 0) && (i + k < imageHeight) && (j + l >= 0) && (j + l < imageWidth))
                    sum += inputImage[img_first_term + j + l] * filter[first_term + l + halffilterSize];

        outputImage[i * imageWidth + j] = sum;

        j += global_size;
        if(j >= imageWidth){
            i += j/imageWidth;
            j %= imageWidth;
        }
    }
}
