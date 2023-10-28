#include "domain_color.cu"

int main() {
  Complex i(0, 1);

  domain_color(
      "identity", [] __device__(Complex z) { return z; }, 0, 1, 2048, 2048);

  domain_color(
      "roots_of_unity", [] __device__(Complex z) { return pow(z, 3) - 1; }, 0,
      2, 2048, 2048);

  domain_color(
      "sqrt", [] __device__(Complex z) { return pow(z, 0.5); }, 0, 2, 2048,
      2048);

  domain_color(
      "removable_singularity", [] __device__(Complex z) { return sin(z) / z; },
      0, M_PI, 2048, 2048);

  domain_color(
      "triple_pole", [] __device__(Complex z) { return 1 / pow(z, 3); }, 0,
      2, 2048, 2048);

  domain_color(
      "essential_singularity", [] __device__(Complex z) { return exp(1 / z); },
      0, 0.5, 2048, 2048);

  domain_color(
      "cluster_point", [] __device__(Complex z) { return tan(1 / z); }, 0, 1,
      2048, 2048);

  domain_color(
      "riemann_zeta",
      [] __device__(Complex z) {
        Complex w = 0;
        for (int i = 1; i < 256; i++) {
          w += pow(i, -z);
        }
        return w;
      },
      0, 7, 2048, 2048);

  domain_color(
      "lacunary",
      [] __device__(Complex z) {
        Complex w = 0;
        for (int i = 0; i < 256; i++) {
          w += z;
          z = pow(z, 2);
        }
        return w;
      },
      0, 1, 2048, 2048);

  domain_color(
      "tetration",
      [] __device__(Complex z) {
        Complex w = z;
        for (int i = 0; i < 31; i++) {
          w = pow(z, w);
        }
        return w;
      },
      0, 3, 2048, 2048);

  domain_color(
      "iterated_carpet",
      [] __device__(Complex z) {
        for (int i = 0; i < 16; i++) {
          z = cos(z) / sin(pow(z, 4) - 1);
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "grace22",
      [] __device__(Complex z) {
        Complex w = z;
        for (int i = 0; i < 22; i++) {
          w = arg(w) * exp(w);
        }
        return w;
      },
      M_PI_2 * i, 0.4, 2048, 2048);

  return 0;
}