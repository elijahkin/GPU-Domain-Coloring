#include "domain_color.cu"

int main() {
  Complex i(0, 1);

  domain_color(
      "identity", [] __device__(Complex z) { return z; }, 0, 1, 2048, 2048);

  domain_color(
      "roots_of_unity", [] __device__(Complex z) { return pow(z, 3) - 1; }, 0,
      2, 2048, 2048);

  // Prototypical example of a branch cut
  // mathworld.wolfram.com/BranchCut.html
  domain_color(
      "sqrt", [] __device__(Complex z) { return sqrt(z); }, 0, 2, 2048,
      2048);

  domain_color(
      "removable_singularity", [] __device__(Complex z) { return sin(z) / z; },
      0, M_PI, 2048, 2048);

  domain_color(
      "triple_pole", [] __device__(Complex z) { return 1 / pow(z, 3); }, 0, 2,
      2048, 2048);

  // Prototypical example of an essential singularity
  // mathworld.wolfram.com/EssentialSingularity.html
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

  // Prototypical example of a natural boundary
  // https://mathworld.wolfram.com/NaturalBoundary.html
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
      "iterated_crevice",
      [] __device__(Complex z) {
        for (int i = 0; i < 16; i++) {
          z = sin(1 / z);
        }
        return z;
      },
      0, 1.07, 2048, 2048);

  domain_color(
      "iterated_spikes",
      [] __device__(Complex z) {
        for (int i = 0; i < 16; i++) {
          z = sin(z) / pow(z, 2);
        }
        return z;
      },
      0, 0.15, 2048, 2048);

  domain_color(
      "iterated_map",
      [] __device__(Complex z) {
        for (int i = 0; i < 16; i++) {
          z = z * sin(pow(z, 3));
        }
        return z;
      },
      0, 2, 2048, 2048);

  // https://en.wikipedia.org/wiki/Newton_fractal
  domain_color(
      "newton1",
      [] __device__(Complex z) {
        for (int i = 0; i < 32; i++) {
          z -= (pow(z, 3) - 1) / (3 * pow(z, 2));
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "newton2",
      [] __device__(Complex z) {
        for (int i = 0; i < 32; i++) {
          z -= (pow(z, 3) + 1) / (3 * pow(z, 2));
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "up_and_coming",
      [] __device__(Complex z) {
        for (int i = 0; i < 8; i++) {
          z = z.imag() * log(z);
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "up_and_coming2",
      [] __device__(Complex z) {
        for (int i = 0; i < 8; i++) {
          z = z.real() * log(z) * exp(i * arg(z));
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "up_and_coming3",
      [] __device__(Complex z) {
        for (int i = 0; i < 64; i++) {
          z = tanh(1 / z) * exp(i * abs(z));
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "up_and_coming4",
      [] __device__(Complex z) {
        for (int i = 0; i < 64; i++) {
          z = z.real() * exp(i * z.imag());
        }
        return z;
      },
      0, 2, 2048, 2048);

  domain_color(
      "up_and_coming5",
      [] __device__(Complex z) {
        for (int i = 0; i < 64; i++) {
          z = z / abs(pow(z, 3) - 1);
        }
        return z * exp(i * arg(sin(pow(z, 3) - 1)));
      },
      0, 1.2, 2048, 2048);

  domain_color(
      "up_and_coming6",
      [] __device__(Complex z) {
        for (int i = 0; i < 64; i++) {
          z = z / (pow(z, 3) - 1).real();
        }
        return z;
      },
      0, 4, 2048, 2048);

  domain_color(
      "up_and_coming7",
      [] __device__(Complex z) {
        for (int i = 0; i < 64; i++) {
          z = pow(z, sqrt(z * z + i));
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