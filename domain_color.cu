#include "utils.cu"

__global__ void domain_color_kernel(Function f, int N, int width, double min_re,
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

    // Convert result to rgb and write to memory
    complex_to_rgb(w, rgb, n);
  }
}

void domain_color(Function f, Complex center, double apothem_real, int width,
                  int height) {
  // Allocate memory for storing pixels
  uint8_t *rgb;
  cudaMallocManaged(&rgb, width * height * 3 * sizeof(uint8_t));

  // Calculate derived constants
  int N = width * height;
  double min_real = center.real() - apothem_real;
  double max_imag = center.imag() + (apothem_real * height) / width;
  double step_size = 2.0 * apothem_real / (width - 1);

  // These blocks and thread numbers were chosen for my RTX 3060
  domain_color_kernel<<<28, 128>>>(f, N, width, min_real, max_imag, step_size,
                                   rgb);

  char filename[100];
  sprintf(filename, "renders/domain_color_%s.png", "happy");
  save_png(rgb, width, height, filename);
}

int main() {
  auto identity = [] __device__(Complex z) -> Complex {
    return z;
  };
  domain_color(identity, 0, 2, 1024, 1024);
  return 0;
}