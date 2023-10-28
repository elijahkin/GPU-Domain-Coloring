#include <nvfunctional>
#include <png.h>
#include <string>
#include <thrust/complex.h>

typedef thrust::complex<double> Complex;
typedef nvstd::function<Complex(Complex)> Function;

void save_png(uint8_t *rgb, int width, int height, std::string name) {
  // Create filename
  name = "renders/" + name + ".png";

  // Open the file
  FILE *fp = fopen(name.c_str(), "wb");

  // Write to the file
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  png_init_io(png, fp);
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  png_bytep row_pointers[height];
  for (int i = 0; i < height; i++) {
    row_pointers[i] = &rgb[i * width * 3];
  }
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
}

__device__ void domain_color_pixel(Complex z, uint8_t *rgb, int n) {
  // Compute the HSL values
  double h = 0.5 * (atan2(z.imag(), z.real()) / M_PI + 1);
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