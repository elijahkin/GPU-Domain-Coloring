#include "domain_color.cu"

int main() {
  Complex i(0, 1);

  auto identity = [] __device__(Complex z) { return z; };
  domain_color("identity", identity, 0, 1, 2048, 2048);

  auto roots_of_unity = [] __device__(Complex z) { return pow(z, 3) - 1; };
  domain_color("roots_of_unity", roots_of_unity, 0, 2, 2048, 2048);

  // Prototypical example of a branch cut
  // mathworld.wolfram.com/BranchCut.html
  auto branch_cut = [] __device__(Complex z) { return sqrt(z); };
  domain_color("branch_cut", branch_cut, 0, 2, 2048, 2048);

  auto removable_singularity = [] __device__(Complex z) { return sin(z) / z; };
  domain_color("removable_singularity", removable_singularity, 0, M_PI, 2048,
               2048);

  auto triple_pole = [] __device__(Complex z) { return 1 / pow(z, 3); };
  domain_color("triple_pole", triple_pole, 0, 2, 2048, 2048);

  // Prototypical example of an essential singularity
  // mathworld.wolfram.com/EssentialSingularity.html
  auto essential_singularity = [] __device__(Complex z) { return exp(1 / z); };
  domain_color("essential_singularity", essential_singularity, 0, 0.5, 2048,
               2048);

  auto cluster_point = [] __device__(Complex z) { return 1 / tan(M_PI / z); };
  domain_color("cluster_point", cluster_point, 0, 1, 2048, 2048);

  auto riemann_zeta = [] __device__(Complex z) {
    Complex w = 0;
    for (int n = 1; n < 256; n++) {
      w += pow(n, -z);
    }
    return w;
  };
  domain_color("riemann_zeta", riemann_zeta, 0, 7, 2048, 2048);

  // Prototypical example of a natural boundary
  // https://mathworld.wolfram.com/NaturalBoundary.html
  auto lacunary = [] __device__(Complex z) {
    Complex w = 0;
    for (int n = 0; n < 256; n++) {
      w += z;
      z *= z;
    }
    return w;
  };
  domain_color("lacunary", lacunary, 0, 1, 2048, 2048);

  auto tetration = [] __device__(Complex z) {
    Complex w = z;
    for (int n = 0; n < 31; n++) {
      w = pow(z, w);
    }
    return w;
  };
  domain_color("tetration", tetration, 0, 3, 2048, 2048);

  auto carpet = [] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = cos(z) / sin(pow(z, 4) - 1);
    }
    return z;
  };
  domain_color("carpet", carpet, 0, 2, 2048, 2048);

  auto crevice = [] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = sin(1 / z);
    }
    return z;
  };
  domain_color("crevice", crevice, 0, 1.07, 2048, 2048);

  auto spikes = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = sin(z) / pow(z, 2);
    }
    return z;
  };
  domain_color("iterated_spikes", spikes, 0, 0.15, 2048, 2048);

  auto iterated_map = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = z * sin(pow(z, 3));
    }
    return z;
  };
  domain_color("iterated_map", iterated_map, 0, 2, 2048, 2048);

  // https://en.wikipedia.org/wiki/Newton_fractal
  auto newton1 = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z -= (pow(z, 3) - 1) / (3 * pow(z, 2));
    }
    return z;
  };
  domain_color("newton1", newton1, 0, 2, 2048, 2048);

  auto newton2 = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z -= (pow(z, 3) + 1) / (3 * pow(z, 2));
    }
    return z;
  };
  domain_color("newton2", newton2, 0, 2, 2048, 2048);

  auto newton3 = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z -= (pow(z, 8) + 15 * pow(z, 4) - 16) / (8 * pow(z, 7) + 60 * pow(z, 3));
    }
    return z;
  };
  domain_color("newton3", newton3, 0, 2, 2048, 2048);

  auto newton4 = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z -= (pow(z, 6) + pow(z, 3) - 1) / (6 * pow(z, 5) + 3 * pow(z, 2));
    }
    return z;
  };
  domain_color("newton4", newton4, 0, 2, 2048, 2048);

  auto newton5 = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z -= (pow(z, 32) - 1) / (32 * pow(z, 31));
    }
    return z;
  };
  domain_color("newton5", newton5, 0, 2, 2048, 2048);

  auto newton6 = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z -= (pow(z, 4) + pow(z, 3) + pow(z, 2) + z + 1) /
           (4 * pow(z, 3) + 3 * pow(z, 2) + 2 * z + 1);
    }
    return z;
  };
  domain_color("newton6", newton6, 0, 2, 2048, 2048);

  auto up_and_coming = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = z.imag() * log(z);
    }
    return z;
  };
  domain_color("up_and_coming", up_and_coming, 0, 2, 2048, 2048);

  auto up_and_coming2 = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = z.real() * log(z) * exp(i * arg(z));
    }
    return z;
  };
  domain_color("up_and_coming2", up_and_coming2, 0, 2, 2048, 2048);

  auto up_and_coming3 = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = tanh(1 / z) * exp(i * abs(z));
    }
    return z;
  };
  domain_color("up_and_coming3", up_and_coming3, 0, 2, 2048, 2048);

  auto up_and_coming4 = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = z.real() * exp(i * z.imag());
    }
    return z;
  };
  domain_color("up_and_coming4", up_and_coming4, 0, 2, 2048, 2048);

  auto up_and_coming5 = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = z / abs(pow(z, 3) - 1);
    }
    return z * exp(i * arg(sin(pow(z, 3) - 1)));
  };
  domain_color("up_and_coming5", up_and_coming5, 0, 1.2, 2048, 2048);

  auto up_and_coming6 = [] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = z / (pow(z, 3) - 1).real();
    }
    return z;
  };
  domain_color("up_and_coming6", up_and_coming6, 0, 4, 2048, 2048);

  auto up_and_coming7 = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = pow(z, sqrt(z * z + i));
    }
    return z;
  };
  domain_color("up_and_coming7", up_and_coming7, 0, 2, 2048, 2048);

  auto grace22 = [] __device__(Complex z) {
    Complex w = z;
    for (int n = 0; n < 22; n++) {
      w = arg(w) * exp(w);
    }
    return w;
  };
  domain_color("grace22", grace22, M_PI_2 * i, 0.4, 2048, 2048);

  auto purple_and_green_slime = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log((pow(z, 6) - i * pow(z, 6) + 7 * z + i * z) /
              (2 * pow(z, 5) + 6));
    }
    return z;
  };
  domain_color("purple_and_green_slime", purple_and_green_slime, 0, 2, 2048,
               2048);

  auto pink_and_green_slime = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = tan(1 / z) * exp(i * abs(z));
    }
    return z;
  };
  domain_color("pink_and_green_slime", pink_and_green_slime, 0, 2.5, 2048,
               2048);

  auto kaleidoscope = [] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = cos(1 / z);
    }
    return log(z);
  };
  domain_color("kaleidoscope", kaleidoscope, 0, 2, 2048, 2048);

  return 0;
}