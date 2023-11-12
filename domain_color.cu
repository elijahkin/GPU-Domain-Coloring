#ifndef utils
#define utils
#include "utils.cu"
#endif

template <typename F>
__device__ Color domain_color_pixel(Complex z, F f) {
  // Apply the function
  Complex w = f(z);

  // Compute the HSL values
  double h = 0.5 * (arg(w) / M_PI + 1);
  double s = 1.0;
  double l = 2.0 * atan(abs(w)) / M_PI; // 0.5;

  // To fix issue with negative real numbers
  h = fmodf(h, 1);

  return hsl_to_rgb(h, s, l);
}

template <typename F>
__global__ void domain_color_kernel(F f, int N, int width, double min_re,
                                    double max_im, double step_size,
                                    uint8_t *rgb) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / width;
    int col = n % width;

    // Convert indices to complex number
    Complex z(min_re + col * step_size, max_im - row * step_size);

    // Map z to a color and store it in memory
    Color color = domain_color_pixel(z, f);
    rgb[3 * n + 0] = color.r;
    rgb[3 * n + 1] = color.g;
    rgb[3 * n + 2] = color.b;
  }
}

template <typename F>
void domain_color(std::string name, F f, Complex center, double apothem_real,
                  int width, int height) {
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
  domain_color_kernel<<<28, 128>>>(f, N, width, min_real, max_imag, step_size,
                                   rgb);
  cudaDeviceSynchronize();

  // Save the image to a file and clean up memory
  name = "renders/domain_color_" + name + ".ppm";
  write_ppm(name, render);
  cudaFree(rgb);
}