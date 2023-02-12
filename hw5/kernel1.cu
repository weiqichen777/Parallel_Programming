#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int resX, int resY, int count, int *img_d) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < resX * resY){

		int thisX = tid % resX; //nth in that row
		int thisY = tid / resX;	//which row

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

		int index = thisY * resX + thisX;
		img_d[index] = i;
	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
// x1, y1, x0, y0, output, width, height, maxIterations
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{

    float stepX = (upperX - lowerX) / resX; //dx
    float stepY = (upperY - lowerY) / resY; //dy


	int block_size = 256;
	int num_blocks = resX * resY / block_size + 1;

	int *output = (int *)malloc(resX * resY * sizeof(int));

	int *img_d;
	cudaMalloc(&img_d, resX * resY * sizeof(int));
	cudaMemcpy(img_d, output, resX * resY * sizeof(int), cudaMemcpyHostToDevice);

	mandelKernel<<<num_blocks, block_size>>>(stepX, stepY, lowerX, lowerY, resX, resY, maxIterations, img_d);

	cudaMemcpy(output, img_d, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<resX*resY; i++){
		img[i] = output[i];
	}

	free(output);
	cudaFree(img_d);

}
