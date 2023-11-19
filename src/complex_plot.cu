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
  domain_color_kernel<<<28, 128>>>(f, render, min_real, max_imag, step_size);
  cudaDeviceSynchronize();

  // Save the image to a file and clean up memory
  name = "renders/domain_color_" + name + ".ppm";
  write_ppm(name, render);
  cudaFree(rgb);
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
  escape_time_kernel<<<28, 128>>>(f, render, min_real, max_imag, step_size,
                                  max_iters);
  cudaDeviceSynchronize();

  // Save the image to a file and clean up memory
  name = "renders/escape_time_" + name + ".ppm";
  write_ppm(name, render);
  cudaFree(rgb);
}

// enum PlotType { domain_color, conformal_map, escape_time };

// template <typename F, typename G>
// __global__ void kernel(F f, Image render, Config config, G extra) {
//   int offset = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;

//   for (int n = offset; n < render.N; n += stride) {
//     // Get this pixel's row and col indices
//     int row = n / render.width;
//     int col = n % render.width;

//     // Convert indices to complex number
//     Complex z(config.min_re + col * config.step_size,
//               config.max_im - row * config.step_size);

//     // Map z to a color and store it in memory
//     Color color;
//     switch (config.type) {
//     case PlotType::domain_color:
//       color = domain_color_pixel(z, f);
//       break;
//     case PlotType::conformal_map:
//       color = conformal_map_pixel(z, f, config, extra);
//       break;
//     case PlotType::escape_time:
//       color = escape_time_pixel(z, f, extra);
//       break;
//     }
//     render.rgb[3 * n + 0] = color.r;
//     render.rgb[3 * n + 1] = color.g;
//     render.rgb[3 * n + 2] = color.b;
//   }
// }

// template <typename F>
// void complex_plot(PlotType type, std::string name, F f, Complex center,
//                   double apothem_real, int width, int height,
//                   std::string pattern_name) {
//   // Calculate derived constants
//   double apothem_imag = (apothem_real * height) / width;
//   double step_size = 2.0 * apothem_real / (width - 1);

//   double min_real = center.real() - apothem_real;
//   double max_real = center.real() + apothem_real;
//   double min_imag = center.imag() - apothem_imag;
//   double max_imag = center.imag() + apothem_imag;

//   // Create the image object to write data to
//   uint8_t *rgb;
//   int N = width * height;
//   cudaMallocManaged(&rgb, 3 * N);
//   Image render = {width, height, N, rgb};

//   // These blocks and thread numbers were chosen for my RTX 3060
//   std::string type_string;
//   switch (type) {
//   case PlotType::domain_color:
//     domain_color_kernel<<<28, 128>>>(f, N, width, min_real, max_imag,
//     step_size,
//                                      rgb);
//     type_string = "domain_color";
//     break;
//   case PlotType::conformal_map:
//     // Read in the pattern
//     Image pattern = read_ppm(pattern_name);
//     conformal_map_kernel<<<28, 128>>>(f, render, min_real, max_real,
//     min_imag,
//                                       max_imag, step_size, pattern);
//     type_string = "conformal_map";
//     break;
//   case PlotType::escape_time:
//     // TODO get max iters from apothem?
//     escape_time_kernel<<<28, 128>>>(f, N, width, min_real, max_imag,
//     step_size,
//                                     rgb, max_iters);
//     type_string = "escape_time";
//     break;
//   }
//   cudaDeviceSynchronize();

//   // Save the image to a file and clean up memory
//   name = "renders/" + type_string + "_" + name + ".ppm";
//   write_ppm(name, render);
//   cudaFree(rgb);
//   cudaFree(pattern.rgb);
// }
