#include <png.h>
#include <string>
#include <thrust/complex.h>

typedef thrust::complex<double> Complex;

void save_ppm(std::string name, uint8_t *rgb, int width, int height) {
  name = "renders/" + name + ".ppm";

  FILE *file = fopen(name.c_str(), "wb");
  if (file) {
    fprintf(file, "P6 %d %d 255\n", width, height);
    fwrite(rgb, 1, 3 * width * height, file);
    fclose(file);
  }
}

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