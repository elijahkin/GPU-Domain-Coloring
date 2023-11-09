#ifndef utils
#define utils
#include "utils.cu"
#endif

struct Image {
  int width;
  int height;
  int N;
  uint8_t *rgb;
};

// enum PlotType { DomainColor, ConformalMap };

// struct PlotConfig {
//   double min_re;
//   double max_re;
//   double min_im;
//   double max_im;
// };

// enum Pattern {
//   Checkerboard = 1,
//   Clock = 2,
//   Lenna = 3,
//   Sirby = 4,
//   Cleo = 5,
//   Eli = 6,
//   Cannon = 7,
//   Blossom = 8,
//   Flower = 9,
// };

Image read_pattern(std::string filename) {
  // Prepare to read in the pattern image
  FILE *fp = fopen(filename.c_str(), "rb");
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  png_init_io(png, fp);
  png_read_info(png, info);

  // Get width and length of pattern
  int width = png_get_image_width(png, info);
  int height = png_get_image_height(png, info);
  int N = width * height;

  // Allocate memory for storing rgb values
  uint8_t *rgb;
  cudaMallocManaged(&rgb, N * 3 * sizeof(uint8_t));

  // Store rgb values in memory
  png_bytep row_pointers[height];
  for (int row = 0; row < height; row++) {
    row_pointers[row] = &rgb[row * width * 3];
  }
  png_read_image(png, row_pointers);

  // Clean up and return the image struct
  fclose(fp);
  png_destroy_read_struct(&png, &info, (png_infopp)NULL);
  return {width, height, N, rgb};
}

__device__ int positive_mod(int a, int b) { return (a % b + b) % b; }

__device__ void conformal_map_pixel(Complex z, uint8_t *rgb, int n,
                                    double min_re, double max_re, double min_im,
                                    double max_im, Image pattern) {
  // Normalize along both the real and imaginary directions
  double re = ((z.real() - min_re) / (max_re - min_re));
  double im = ((max_im - z.imag()) / (max_im - min_im));

  // Map to pixel coordinates in the input image
  int img_col = floor(re * pattern.width);
  int img_row = floor(im * pattern.height);
  
  // Wrap around the indices which are out of bounds
  img_col = positive_mod(img_col, pattern.width);
  img_row = positive_mod(img_row, pattern.height);
  
  // Find the corresponding pixel in the input image
  int m = img_row * pattern.width + img_col;

  // Write its RGB values to memory
  rgb[3 * n + 0] = pattern.rgb[3 * m + 0];
  rgb[3 * n + 1] = pattern.rgb[3 * m + 1];
  rgb[3 * n + 2] = pattern.rgb[3 * m + 2];
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

    // Apply the function
    Complex w = f(z);

    conformal_map_pixel(w, render.rgb, n, min_re, max_re, min_im, max_im, pattern);
  }
}

template <typename F>
void conformal_map(std::string name, F f, Complex center, double apothem_real,
                   int width, int height, std::string pattern_name) {
  // Invoke the loader
  Image pattern = read_pattern(pattern_name);

  // Allocate memory for storing pixels
  uint8_t *rgb;
  cudaMallocManaged(&rgb, width * height * 3 * sizeof(uint8_t));

  // Calculate derived constants
  int N = width * height;
  double apothem_imag = (apothem_real * height) / width;
  double step_size = 2.0 * apothem_real / (width - 1);

  double min_real = center.real() - apothem_real;
  double max_real = center.real() + apothem_real;
  double min_imag = center.imag() - apothem_imag;
  double max_imag = center.imag() + apothem_imag;

  Image render = {width, height, N, rgb};

  // These blocks and thread numbers were chosen for my RTX 3060
  conformal_map_kernel<<<28, 128>>>(f, render, min_real, max_real, min_imag,
                                    max_imag, step_size, pattern);

  // Wait for all threads to finish
  cudaDeviceSynchronize();
  cudaFree(pattern.rgb);

  // Save the image to a file
  name = "conformal_map_" + name;
  save_ppm(name, rgb, width, height);
  cudaFree(rgb);
}