#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int resX, int resY, int count, int *img_d, int pitch, int group_size) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < resX*resY/group_size){

		int my_start = tid * group_size;
		int my_end = my_start + group_size;

		for(int j=my_start; j<my_end; j++){

			int thisX = j % resX;	//nth in that row
			int thisY = j / resX;	//which row
			float x = lowerX + thisX * stepX;
			float y = lowerY + thisY * stepY;
			float z_re = x, z_im = y;

			int i;
			for(i=0; i<count; i++){

				if(z_re * z_re + z_im * z_im > 4.f){
					break;
				}

				float new_re = z_re * z_re - z_im * z_im;
				float new_im = 2.f * z_re * z_im;
				z_re = x + new_re;
				z_im = y + new_im;
			}

			// int index = thisY * resX + thisX;
			//img_d[index] = i;

			int *rowHead;
			rowHead = (int *)((char *)img_d + thisY * pitch);
			rowHead[thisX] = i;
		}

	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
// x1, y1, x0, y0, output, width, height, maxIterations
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX; //dx
    float stepY = (upperY - lowerY) / resY; //dy

    int group_size = 16;
	int block_size = 256;
	int num_blocks = (resX * resY) / (block_size * group_size) + 1;

	int *output;
	cudaHostAlloc(&output, resX * resY * sizeof(int), cudaHostAllocDefault);

	size_t pitch;
	int *img_d;
	cudaMallocPitch(&img_d, &pitch, resX * sizeof(int), resY);

	cudaMemcpy2D(img_d, pitch, output, resX * sizeof(int), resX * sizeof(int), resY, cudaMemcpyHostToDevice);

	mandelKernel<<<num_blocks, block_size>>>(stepX, stepY, lowerX, lowerY, resX, resY, maxIterations, img_d, pitch, group_size);

	cudaMemcpy2D(output, resX * sizeof(int), img_d, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

	for(int i=0; i<resX*resY; i++){
		img[i] = output[i];
	}

	cudaFreeHost(output);
	cudaFree(img_d);
}
