/*******************************************************************************
To compile: gcc -O3 -o mandelbrot mandelbrot.c -lm
To create an image with 4096 x 4096 pixels: ./mandelbrot 4096 4096
*******************************************************************************/
/*Convert the Serial code to Parallel with GPU using CUDA. Parallelizing mandelbrot function
and converting the testpoint function to a device function*/
/*******************************************************************************/
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI);

#define MXITER 1000

/*******************************************************************************/
// Define a complex number
typedef struct {
  double x;
  double y;
}complex_t;


/*******************************************************************************/
// Return iterations before z leaves mandelbrot set for given c
__device__ int testpoint(complex_t c){
  int iter;  
  complex_t z = c;

  for(iter=0; iter<MXITER; iter++){ 
    // real part of z^2 + c 
    double tmp = (z.x*z.x) - (z.y*z.y) + c.x;
    // update with imaginary part of z^2 + c
    z.y = z.x*z.y*2. + c.y;
    // update real part
    z.x = tmp; 
    // check bound
    if((z.x*z.x+z.y*z.y)>4.0){ return iter;}
  }
  return iter; 
}

/*******************************************************************************/
// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array
__global__ void mandelbrotKernel(int Nre, int Nim, complex_t cmin, complex_t dc, float *count) {
    // computing the position of the current thread in the grid
    int m = blockIdx.x * blockDim.x + threadIdx.x; // X coordinate in the grid
    int n = blockIdx.y * blockDim.y + threadIdx.y; // Y coordinate in the grid

    if (m < Nre && n < Nim) {
        complex_t c;
        c.x = cmin.x + dc.x * m; // Calculate the real part of 'c'
        c.y = cmin.y + dc.y * n; // Calculate the imaginary part of 'c'

        int iteration = testpoint(c); // Calculate how many iterations it takes for 'c' to escape

        // Calculate the index in the array
        int index = n * Nre + m;
        count[index] = (float)iteration; // Store the result
    }
}

/*******************************************************************************/
int main(int argc, char **argv) {
    int Nre = (argc == 3) ? atoi(argv[1]) : 4096;
    int Nim = (argc == 3) ? atoi(argv[2]) : 4096;

    // Host storage for the iteration counts
    float *count = (float*) malloc(Nre * Nim * sizeof(float));

    // Parameters for a bounding box for "c" that generates an interesting image
    const float centRe = -0.5, centIm = 0;
    const float diam = 3.0;

    complex_t cmin; 
    complex_t cmax;
    complex_t dc;

    cmin.x = centRe - 0.5 * diam;
    cmax.x = centRe + 0.5 * diam;
    cmin.y = centIm - 0.5 * diam;
    cmax.y = centIm + 0.5 * diam;

    // Set step sizes
    dc.x = (cmax.x - cmin.x) / (Nre - 1);
    dc.y = (cmax.y - cmin.y) / (Nim - 1);

    // Allocate memory on the device
    float *d_count;
    cudaMalloc((void**)&d_count, Nre * Nim * sizeof(float));

    // Kernel execution configuration
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((Nre + blockSize.x - 1) / blockSize.x, (Nim + blockSize.y - 1) / blockSize.y);

    // Declare CUDA Event for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    // Compute Mandelbrot set
    mandelbrotKernel<<<gridSize, blockSize>>>(Nre, Nim, cmin, dc, d_count);
    cudaDeviceSynchronize(); // Ensure kernel completion

    // Copy from the GPU back to the host here
    cudaMemcpy(count, d_count, Nre * Nim * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f seconds\n", milliseconds / 1000.0);

    // Output Mandelbrot to PPM format image
    printf("Printing mandelbrot.ppm...\n");
    writeMandelbrot("GPU_mandelbrot.ppm", Nre, Nim, count, 0, 80);
    printf("done.\n");

    // Free resources
    cudaFree(d_count);
    free(count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


/* Output data as PPM file */
void saveppm(const char *filename, unsigned char *img, int width, int height){

  /* FILE pointer */
  FILE *f;
  
  /* Open file for writing */
  f = fopen(filename, "wb");
  
  /* PPM header info, including the size of the image */
  fprintf(f, "P6 %d %d %d\n", width, height, 255);

  /* Write the image data to the file - remember 3 byte per pixel */
  fwrite(img, 3, width*height, f);

  /* Make sure you close the file */
  fclose(f);
}



int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI){

  int n, m;
  unsigned char *rgb   = (unsigned char*) calloc(3*width*height, sizeof(unsigned char));
  
  for(n=0;n<height;++n){
    for(m=0;m<width;++m){
      int id = m+n*width;
      int I = (int) (768*sqrt((double)(img[id]-minI)/(maxI-minI)));
      
      // change this to change palette
      if(I<256)      rgb[3*id+2] = 255-I;
      else if(I<512) rgb[3*id+1] = 511-I;
      else if(I<768) rgb[3*id+0] = 767-I;
      else if(I<1024) rgb[3*id+0] = 1023-I;
      else if(I<1536) rgb[3*id+1] = 1535-I;
      else if(I<2048) rgb[3*id+2] = 2047-I;

    }
  }

  saveppm(fileName, rgb, width, height);

  free(rgb);
  return 0;
}