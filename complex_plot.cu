#include "domain_color.cu"

int main() {
  domain_color([] __device__(Complex z) { return z; }, 0, 1, 2048, 2048,
               "identity");

  domain_color([] __device__(Complex z) { return pow(z, 3) - 1; }, 0, 2, 2048,
               2048, "roots_of_unity");

  domain_color([] __device__(Complex z) { return pow(z, 0.5); }, 0, 2, 2048,
               2048, "sqrt");

  domain_color([] __device__(Complex z) { return sin(z) / z; }, 0, M_PI, 2048,
               2048, "removable_singularity");

  domain_color([] __device__(Complex z) { return 1 / pow(z, 3); }, 0, 2, 2048,
               2048, "pole_of_order_3");

  domain_color([] __device__(Complex z) { return exp(1 / z); }, 0, 0.5, 2048,
               2048, "essential_singularity");

  domain_color([] __device__(Complex z) { return tan(1 / z); }, 0, 1, 2048,
               2048, "cluster_point");

  domain_color(
      [] __device__(Complex z) {
        Complex w = 0;
        for (int i = 1; i < 1024; i++) {
          w += pow(i, -z);
        }
        return w;
      },
      0, 7, 2048, 2048, "riemann_zeta");
  return 0;

  domain_color(
      [] __device__(Complex z) {
        Complex w = 0;
        for (int i = 0; i < 256; i++) {
          w += z;
          z = pow(z, 2);
        }
        return w;
      },
      0, 1, 2048, 2048, "lacunary");
  return 0;
}