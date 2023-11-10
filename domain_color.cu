#ifndef utils
#define utils
#include "utils.cu"
#endif

__device__ void domain_color_pixel(Complex z, uint8_t *rgb, int n) {
  // Compute the HSL values
  double h = 0.5 * (arg(z) / M_PI + 1);
  double s = 1.0;
  double l = 2.0 * atan(abs(z)) / M_PI;

  // Convert HSL values to RGB values
  double chroma = (1 - abs(2 * l - 1)) * s;
  double x = chroma * (1 - abs(fmod(6 * h, 2.0) - 1));
  double m = l - (chroma / 2.0);

  uint8_t sextant = int(h * 6);
  double r, g, b;
  switch (sextant) {
  case 0:
    r = chroma;
    g = x;
    b = 0;
    break;
  case 1:
    r = x;
    g = chroma;
    b = 0;
    break;
  case 2:
    r = 0;
    g = chroma;
    b = x;
    break;
  case 3:
    r = 0;
    g = x;
    b = chroma;
    break;
  case 4:
    r = x;
    g = 0;
    b = chroma;
    break;
  case 5:
    r = chroma;
    g = 0;
    b = x;
    break;
  default:
    r = g = b = 0;
    break;
  }

  // Write RGB values to memory
  rgb[3 * n + 0] = (uint8_t)((r + m) * 255);
  rgb[3 * n + 1] = (uint8_t)((g + m) * 255);
  rgb[3 * n + 2] = (uint8_t)((b + m) * 255);
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

    // Apply the function
    Complex w = f(z);

    // Convert the value to rgb and store it in memory
    domain_color_pixel(w, rgb, n);
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