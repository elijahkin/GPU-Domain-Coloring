#include <cstdint>

struct Color {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

__device__ Color hsl_to_rgb(double h, double s, double l) {
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
  return {(uint8_t)((r + m) * 255), (uint8_t)((g + m) * 255),
          (uint8_t)((b + m) * 255)};
}

__device__ uint8_t lerp(uint8_t a, uint8_t b, double t) {
  return static_cast<uint8_t>(a + t * (b - a));
}

__device__ Color color_lerp(Color x, Color y, double t) {
  return {lerp(x.r, y.r, t), lerp(x.g, y.g, t), lerp(x.b, y.b, t)};
}
