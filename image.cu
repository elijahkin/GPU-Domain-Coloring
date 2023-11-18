#include <string>

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