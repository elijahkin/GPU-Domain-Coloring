#include <iostream>
#include <png.h>

// TODO Eventually some of these should be command line args

// Configure dimensions of output image
const int kWidth = 8192;
const int kHeight = 6144;

// Configure subset of the complex plane to graph
const double center_re = -0.7;
const double center_im = 0;
const double apothem_re = 1.6;

// Compute derived constant
const double apothem_im = (apothem_re * kHeight) / kWidth;
const double min_re = center_re - apothem_re;
const double max_im = center_im + apothem_im;
const double step_size = 2.0 * apothem_re / (kWidth - 1);

// Do all Mandelbrot related math on the GPU
__global__ void mandelbrot_cuda(int N, uint8_t *rgb) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / kWidth;
    int col = n % kWidth;

    // Initialize c and z variables
    double c_re = min_re + col * step_size;
    double c_im = max_im - row * step_size;
    double z_re = 0.0;
    double z_im = 0.0;

    // Perform Mandelbrot iterations
    int iter;
    for (iter = 0; iter < 256; iter++) {
      double tmp = fma(z_re, z_re, -fma(z_im, z_im, -c_re));
      z_im = fma(z_re, z_im, fma(z_re, z_im, c_im));
      z_re = tmp;

      // End iterating if z escapes the circle of radius 2
      if (z_re * z_re + z_im * z_im > 4) {
        break;
      }
    }

    // Convert iteration number to HSL
    // TODO

    // Convert HSL to RGB
    // TODO

    // Write RGB values to memory
    rgb[3 * n + 0] = 0;
    rgb[3 * n + 1] = iter;
    rgb[3 * n + 2] = 0;
  }
}

void save_png(uint8_t *rgb, char *filename) {
  FILE *fp = fopen(filename, "wb");
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  png_init_io(png, fp);
  png_set_IHDR(png, info, kWidth, kHeight, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  png_bytep row_pointers[kHeight];
  for (int i = 0; i < kHeight; i++) {
    row_pointers[i] = &rgb[i * kWidth * 3];
  }
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
}

int main() {
  int N = kHeight * kWidth;

  // Allocate memory
  uint8_t *rgb;
  cudaMallocManaged(&rgb, N * 3 * sizeof(uint8_t));

  // These blocks and thread numbers were chosen for my RTX 3060
  mandelbrot_cuda<<<28, 128>>>(N, rgb);

  // Wait for all threads to finish
  cudaDeviceSynchronize();

  // Save image
  char filename[100];
  sprintf(filename, "renders/mandelbrot_%i_%i.png", kWidth, kHeight);
  save_png(rgb, filename);

  // Deallocate memory
  cudaFree(rgb);
  return 0;
}