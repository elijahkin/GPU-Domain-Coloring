#include "utils.cu"

// struct Image {
//   int width;
//   int height;
//   int N;
//   uint8_t *rgb;
// };

__global__ void conformal_map_kernel(Function f, int N, int width,
                                     double min_re, double max_re,
                                     double min_im, double max_im,
                                     double step_size, uint8_t *rgb,
                                     uint8_t *img, int img_width,
                                     int img_height) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / width;
    int col = n % width;

    // Convert indices to complex number
    Complex z(min_re + col * step_size, max_im - row * step_size);

    // Apply the function
    Complex w = essential_singularity(z);

    // TODO Can we put everything below in a __device__ function?
    // Normalize along both the real and imaginary directions
    double re = ((w.real() - min_re) / (max_re - min_re));
    double im = ((max_im - w.imag()) / (max_im - min_im));

    // Map to pixel coordinates in the input image
    int img_col = re * img_width;
    int img_row = im * img_height;

    // Wrap around the indices which are out of bounds
    img_col %= img_width;
    img_row %= img_height;

    // Find the corresponding pixel in the input image
    int m = img_row * img_width + img_col;

    // Write its RGB values to memory
    rgb[3 * n + 0] = img[3 * m + 0];
    rgb[3 * n + 1] = img[3 * m + 1];
    rgb[3 * n + 2] = img[3 * m + 2];
  }
}

void conformal_map(Function f, Complex center, double apothem_real, int width,
                   int height, std::string input, std::string name) {
  // Compute dimensions of input image
  int img_width = ;
  int img_height = ;

  // Allocate memory for input image
  uint8_t *img;
  cudaMallocManaged(&img, img_width * img_height * 3 * sizeof(uint8_t));

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

  // These blocks and thread numbers were chosen for my RTX 3060
  conformal_map_kernel<<<28, 128>>>(TODO);

  // Wait for all threads to finish
  cudaDeviceSynchronize();

  // Save the image to a file
  name = "conformal_map_" + name;
  save_png(rgb, width, height, name);
}