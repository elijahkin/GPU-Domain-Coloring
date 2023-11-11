#ifndef utils
#define utils
#include "utils.cu"
#endif

template <typename F>
__device__ void escape_time_pixel(Complex c, F f, int max_iters, uint8_t *rgb,
                                  int n) {
  // Perform Mandelbrot iterations
  int iter;
  Complex z = 0;

  for (iter = 0; iter < max_iters; iter++) {
    z = f(z, c);

    // End iterating if z escapes the circle of radius 2
    if (abs(z) > 2) {
      break;
    }
  }

  // Convert iteration number to HSL
  // TODO

  // Convert HSL to RGB
  // TODO

  // Write RGB values to memory
  rgb[3 * n + 0] = 255 * (1 - static_cast<double>(iter) / max_iters);
  rgb[3 * n + 1] = 255 * (1 - static_cast<double>(iter) / max_iters);
  rgb[3 * n + 2] = 255;
}

template <typename F>
__global__ void escape_time_kernel(F f, int N, int width, double min_re,
                                   double max_im, double step_size,
                                   uint8_t *rgb, int max_iters) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / width;
    int col = n % width;

    // Convert indices to complex number
    Complex z(min_re + col * step_size, max_im - row * step_size);

    // Convert the value to rgb and store it in memory
    escape_time_pixel(z, f, max_iters, rgb, n);
  }
}

template <typename F>
void escape_time(std::string name, F f, Complex center, double apothem_real,
                 int width, int height, int max_iters) {
  // Calculate derived constants
  int N = width * height;
  double min_real = center.real() - apothem_real;
  double max_imag = center.imag() + (apothem_real * height) / width;
  double step_size = 2.0 * apothem_real / (width - 1);

  // Create the image object to write data to
  uint8_t *rgb;
  cudaMallocManaged(&rgb, width * height * 3 * sizeof(uint8_t));
  Image render = {width, height, N, rgb};

  // These blocks and thread numbers were chosen for my RTX 3060
  escape_time_kernel<<<28, 128>>>(f, N, width, min_real, max_imag, step_size,
                                  rgb, max_iters);
  cudaDeviceSynchronize();

  // Save the image to a file and clean up memory
  name = "renders/escape_time_" + name + ".ppm";
  write_ppm(name, render);
  cudaFree(rgb);
}