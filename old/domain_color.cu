#include <iostream>
#include <png.h>
#include <thrust/complex.h>

// Configure dimensions of output image
const int kWidth = 8192;
const int kHeight = 8192;

// Configure subset of the complex plane to graph
const double center_re = 0;
const double center_im =  1.5707;
const double apothem_re = 0.4;

// Compute derived constant
const double apothem_im = (apothem_re * kHeight) / kWidth;
const double min_re = center_re - apothem_re;
const double max_im = center_im + apothem_im;
const double step_size = 2.0 * apothem_re / (kWidth - 1);

void save_png(uint8_t *rgb, char *filename) {
  FILE *fp = fopen(filename, "wb");
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  png_init_io(png, fp);
  png_set_IHDR(png, info, kWidth, kHeight, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  png_bytep row_pointers[kHeight];
  for (int i = 0; i < kHeight; i++) {
    row_pointers[i] = &rgb[i * kWidth * 3];
  }
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
}

__device__ thrust::complex<double> identity(thrust::complex<double> z) {
    return z;
}

__device__ thrust::complex<double> essential_singularity(thrust::complex<double> z) {
    return exp(1.0 / z);
}

__device__ thrust::complex<double> grace(thrust::complex<double> z) {
  thrust::complex<double> w = z;
  for (int i = 0; i < 22; i++) {
    w = arg(w) * exp(w);
  }
  return w;
}

__device__ thrust::complex<double> grace2(thrust::complex<double> z) {
  thrust::complex<double> w = z;
  for (int i = 0; i < 22; i++) {
    w = (pow(w, 4) + w - 1) / (4 * pow(w, 3) + 1);
  }
  return w;
}

__global__ void domain_color_cuda(int N, uint8_t *rgb) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int n = offset; n < N; n += stride) {
    // Get this pixel's row and col indices
    int row = n / kWidth;
    int col = n % kWidth;

    // Convert indices to complex number
    thrust::complex<double> z(min_re + col * step_size,
                              max_im - row * step_size);

    // Apply the function
    thrust::complex<double> w = grace(z);

    // Compute the HSL values
    double h = 0.5 * (atan2(w.imag(), w.real()) / M_PI + 1);
    double s = 1.0;
    double l = 2.0 * atan(abs(w)) / M_PI;

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
    rgb[3 * n + 0] = (uint8_t) ((r + m) * 255);
    rgb[3 * n + 1] = (uint8_t) ((g + m) * 255);
    rgb[3 * n + 2] = (uint8_t) ((b + m) * 255);
  }
}

void domain_color() {
  int N = kHeight * kWidth;

  // Allocate memory
  uint8_t *rgb;
  cudaMallocManaged(&rgb, N * 3 * sizeof(uint8_t));

  // These blocks and thread numbers were chosen for my RTX 3060
  domain_color_cuda<<<28, 128>>>(N, rgb);

  // Wait for all threads to finish
  cudaDeviceSynchronize();

  // Save image
  char filename[100];
  sprintf(filename, "renders/domain_color_%i_%i.png", kWidth, kHeight);
  save_png(rgb, filename);

  // Deallocate memory
  cudaFree(rgb);
}

int main() {
  domain_color();
  return 0;
}