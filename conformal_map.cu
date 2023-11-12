#ifndef utils
#define utils
#include "utils.cu"
#endif

__device__ int positive_mod(int a, int b) { return (a % b + b) % b; }

template <typename F>
__device__ Color conformal_map_pixel(Complex z, F f, double min_re,
                                     double max_re, double min_im,
                                     double max_im, Image pattern) {
  Complex w = f(z);

  // Normalize along both the real and imaginary directions
  double re = ((w.real() - min_re) / (max_re - min_re));
  double im = ((max_im - w.imag()) / (max_im - min_im));

  // Map to pixel coordinates in the input image
  int img_col = floor(re * pattern.width);
  int img_row = floor(im * pattern.height);

  // Wrap around the indices which are out of bounds
  img_col = positive_mod(img_col, pattern.width);
  img_row = positive_mod(img_row, pattern.height);

  // Find the corresponding pixel in the input image
  int m = img_row * pattern.width + img_col;

  // Write its RGB values to memory
  return {pattern.rgb[3 * m + 0], pattern.rgb[3 * m + 1],
          pattern.rgb[3 * m + 2]};
}

template <typename F>
__global__ void conformal_map_kernel(F f, Image render, double min_re,
                                     double max_re, double min_im,
                                     double max_im, double step_size,
                                     Image pattern) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < render.N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / render.width;
    int col = n % render.width;

    // Convert indices to complex number
    Complex z(min_re + col * step_size, max_im - row * step_size);

    // Map z to a color and store it in memory
    Color color =
        conformal_map_pixel(z, f, min_re, max_re, min_im, max_im, pattern);
    render.rgb[3 * n + 0] = color.r;
    render.rgb[3 * n + 1] = color.g;
    render.rgb[3 * n + 2] = color.b;
  }
}

template <typename F>
void conformal_map(std::string name, F f, Complex center, double apothem_real,
                   int width, int height, std::string pattern_name) {
  // Calculate derived constants
  int N = width * height;
  double apothem_imag = (apothem_real * height) / width;
  double step_size = 2.0 * apothem_real / (width - 1);

  double min_real = center.real() - apothem_real;
  double max_real = center.real() + apothem_real;
  double min_imag = center.imag() - apothem_imag;
  double max_imag = center.imag() + apothem_imag;

  // Read in the pattern
  Image pattern = read_ppm(pattern_name);

  // Create the image object to write data to
  uint8_t *rgb;
  cudaMallocManaged(&rgb, width * height * 3 * sizeof(uint8_t));
  Image render = {width, height, N, rgb};

  // These blocks and thread numbers were chosen for my RTX 3060
  conformal_map_kernel<<<28, 128>>>(f, render, min_real, max_real, min_imag,
                                    max_imag, step_size, pattern);
  cudaDeviceSynchronize();

  // Save the image to a file and clean up memory
  name = "renders/conformal_map_" + name + ".ppm";
  write_ppm(name, render);
  cudaFree(rgb);
  cudaFree(pattern.rgb);
}