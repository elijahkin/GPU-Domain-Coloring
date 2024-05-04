#include "color.cu"
#include "image.cu"

#include <thrust/complex.h>
typedef thrust::complex<double> Complex;

template <typename F> __device__ Color domain_color_pixel(Complex z, F f) {
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

  // // TODO replace this above with this
  // int img_col = floor((1 + w.real()) * pattern.width * 0.5);
  // int img_row = floor((1 + w.imag()) * pattern.height * -0.5);

  // Wrap around the indices which are out of bounds
  img_col = positive_mod(img_col, pattern.width);
  img_row = positive_mod(img_row, pattern.height);

  // Find the corresponding pixel in the input image
  int m = img_row * pattern.width + img_col;

  // Write its RGB values to memory
  return {pattern.rgb[3 * m + 0], pattern.rgb[3 * m + 1],
          pattern.rgb[3 * m + 2]};
}

__device__ Color iter_to_color(int iter, int max_iters) {
  double h =
      fmodf(pow((static_cast<double>(iter) / max_iters) * 360, 1.5), 360) /
      360.0;
  return hsl_to_rgb(h, 0.5, 0.5);
}

template <typename F>
__device__ Color escape_time_pixel(Complex c, F f, int max_iters) {
  int iter;
  Complex z = 0;

  for (iter = 0; iter < max_iters; iter++) {
    z = f(z, c);

    // End iterating if z escapes the circle of radius 2
    if (abs(z) > 2) {
      break;
    }
  }

  double smooth_iter = iter + 1 - log2(log2(abs(z)));

  Color color1 = iter_to_color((int)(smooth_iter), max_iters);
  Color color2 = iter_to_color((int)(smooth_iter + 1), max_iters);

  return color_lerp(color1, color2, fmodf(smooth_iter, 1));
}

template <typename F>
__global__ void domain_color_kernel(F f, Image render, double min_re,
                                    double max_im, double step_size) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < render.N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / render.width;
    int col = n % render.width;

    // Convert indices to complex number
    Complex z(min_re + col * step_size, max_im - row * step_size);

    // Map z to a color and store it in memory
    Color color = domain_color_pixel(z, f);
    render.rgb[3 * n + 0] = color.r;
    render.rgb[3 * n + 1] = color.g;
    render.rgb[3 * n + 2] = color.b;
  }
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
__global__ void escape_time_kernel(F f, Image render, double min_re,
                                   double max_im, double step_size,
                                   int max_iters) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < render.N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / render.width;
    int col = n % render.width;

    // Convert indices to complex number
    Complex z(min_re + col * step_size, max_im - row * step_size);

    // Map z to a color and store it in memory
    Color color = escape_time_pixel(z, f, max_iters);
    render.rgb[3 * n + 0] = color.r;
    render.rgb[3 * n + 1] = color.g;
    render.rgb[3 * n + 2] = color.b;
  }
}
