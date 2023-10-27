#include "utils.cu"

struct Image {
  int width;
  int height;
  int N;
  uint8_t *rgb;
};

__global__ void conformal_map_kernel(Function f, int N, int width,
                                     double min_re, double max_im,
                                     double step_size, uint8_t *rgb,
                                     uint8_t *in) {
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
    double re = ((w.real() - (center.real - r)) / (2 * r));
    double im = ((w.imag() - (center.imag - r)) / (2 * r));

    // Map to pixel coordinates in the input image
    int p = re * in.width;
    int q = im * in.height;

    // Wrap around the indices which are out of bounds
    p %= in.width;
    q %= in.height;

    // Find the corresponding pixel in the input image
    int m = p * in.width + q;

    // Write its RGB values to memory
    out.rgb[3 * n + 0] = in.rgb[3 * m + 0];
    out.rgb[3 * n + 1] = in.rgb[3 * m + 1];
    out.rgb[3 * n + 2] = in.rgb[3 * m + 2];
  }
}

void conformal_map(int width, int height) {}