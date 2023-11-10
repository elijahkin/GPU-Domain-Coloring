#include <string>
#include <thrust/complex.h>

typedef thrust::complex<double> Complex;

struct Image {
  int width;
  int height;
  int N;
  uint8_t *rgb;
};

Image read_ppm(std::string filename) {
  FILE *file = fopen(filename.c_str(), "r");
  Image image;
  if (file) {
    fscanf(file, "P6\n %d %d\n 255\n", &image.width, &image.height);
    image.N = image.width * image.height;
    cudaMallocManaged(&image.rgb, 3 * image.N);
    fread(image.rgb, 1, 3 * image.N, file);
    fclose(file);
  }
  return image;
}

void write_ppm(std::string filename, const Image &image) {
  FILE *file = fopen(filename.c_str(), "wb");
  if (file) {
    fprintf(file, "P6\n %d %d\n 255\n", image.width, image.height);
    fwrite(image.rgb, 1, 3 * image.N, file);
    fclose(file);
  }
}

// enum PlotType { DomainColor, ConformalMap };

// struct PlotConfig {
//   PlotType type;
//   double min_re;
//   double max_re;
//   double min_im;
//   double max_im;
//   double step_size;
// };

// __device__ void domain_color_pixel(Complex z, uint8_t *rgb, int n) {
//   // Compute the HSL values
//   double h = 0.5 * (arg(z) / M_PI + 1);
//   double s = 1.0;
//   double l = 2.0 * atan(abs(z)) / M_PI;

//   // Convert HSL values to RGB values
//   double chroma = (1 - abs(2 * l - 1)) * s;
//   double x = chroma * (1 - abs(fmod(6 * h, 2.0) - 1));
//   double m = l - (chroma / 2.0);

//   uint8_t sextant = int(h * 6);
//   double r, g, b;
//   switch (sextant) {
//   case 0:
//     r = chroma;
//     g = x;
//     b = 0;
//     break;
//   case 1:
//     r = x;
//     g = chroma;
//     b = 0;
//     break;
//   case 2:
//     r = 0;
//     g = chroma;
//     b = x;
//     break;
//   case 3:
//     r = 0;
//     g = x;
//     b = chroma;
//     break;
//   case 4:
//     r = x;
//     g = 0;
//     b = chroma;
//     break;
//   case 5:
//     r = chroma;
//     g = 0;
//     b = x;
//     break;
//   default:
//     r = g = b = 0;
//     break;
//   }

//   // Write RGB values to memory
//   rgb[3 * n + 0] = (uint8_t)((r + m) * 255);
//   rgb[3 * n + 1] = (uint8_t)((g + m) * 255);
//   rgb[3 * n + 2] = (uint8_t)((b + m) * 255);
// }

// __device__ int positive_mod(int a, int b) { return (a % b + b) % b; }

// __device__ void conformal_map_pixel(Complex z, uint8_t *rgb, int n,
//                                     PlotConfig config, Image pattern) {
//   // Normalize along both the real and imaginary directions
//   double re = ((z.real() - config.min_re) / (config.max_re - config.min_re));
//   double im = ((config.max_im - z.imag()) / (config.max_im - config.min_im));

//   // Map to pixel coordinates in the input image
//   int img_col = floor(re * pattern.width);
//   int img_row = floor(im * pattern.height);

//   // Wrap around the indices which are out of bounds
//   img_col = positive_mod(img_col, pattern.width);
//   img_row = positive_mod(img_row, pattern.height);

//   // Find the corresponding pixel in the input image
//   int m = img_row * pattern.width + img_col;

//   // Write its RGB values to memory
//   rgb[3 * n + 0] = pattern.rgb[3 * m + 0];
//   rgb[3 * n + 1] = pattern.rgb[3 * m + 1];
//   rgb[3 * n + 2] = pattern.rgb[3 * m + 2];
// }

// template <typename F>
// __global__ void kernel(F f, Image render, PlotConfig config, Image pattern) {
//   int offset = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;

//   for (int n = offset; n < render.N; n += stride) {
//     // Get this pixel's row and col indices
//     int row = n / render.width;
//     int col = n % render.width;

//     // Convert indices to complex number
//     Complex z(config.min_re + col * config.step_size,
//               config.max_im - row * config.step_size);

//     // Apply the function
//     Complex w = f(z);

//     switch (config.type) {
//     case PlotType::DomainColor:
//       domain_color_pixel(w, render.rgb, n);
//       break;
//     case PlotType::ConformalMap:
//       conformal_map_pixel(w, render.rgb, n, config, pattern);
//       break;
//     }
//   }
// }

// template <typename F>
// void complex_plot(PlotType type, std::string name, F f, Complex center,
//                   double apothem_real, int width, int height,
//                   std::string pattern_name) {
//   // Calculate derived constants
//   double apothem_imag = (apothem_real * height) / width;
//   double step_size = 2.0 * apothem_real / (width - 1);
//   PlotConfig config = {type,
//                        center.real() - apothem_real,
//                        center.real() + apothem_real,
//                        center.imag() - apothem_imag,
//                        center.imag() + apothem_imag,
//                        step_size};

//   // Read in the pattern
//   Image pattern = read_ppm(pattern_name);

//   // Create the image object to write data to
//   uint8_t *rgb;
//   int N = width * height;
//   cudaMallocManaged(&rgb, 3 * N);
//   Image render = {width, height, N, rgb};

//   // These blocks and thread numbers were chosen for my RTX 3060
//   kernel<<<28, 128>>>(f, render, config, pattern);
//   cudaDeviceSynchronize();

//   // Save the image to a file and clean up memory
//   name = "renders/newguy_" + name + ".ppm";
//   write_ppm(name, render);
//   cudaFree(rgb);
//   cudaFree(pattern.rgb);
// }