#include "conformal_map.cu"
#include "domain_color.cu"

int main() {
  Complex i(0, 1);

  auto identity = [] __device__(Complex z) { return z; };
  domain_color("identity", identity, 0, 1, 2048, 2048);
  conformal_map("identity", identity, 0, 1, 2048, 2048,
                "patterns/checkerboard.png");

  auto shear = [i] __device__(Complex z) { return 3 * z * exp(i * M_PI_4); };
  conformal_map("shear", shear, 0, 1, 2048, 2048, "patterns/checkerboard.png");

  auto inverse = [] __device__(Complex z) { return 1 / z; };
  conformal_map("blossom_inverse", inverse, 0, M_PI / 8, 2048, 2048,
                "patterns/blossom.png");
  conformal_map("longeli_inverse", inverse, 0, 1, 2048, 2048,
                "patterns/longeli.png");
  conformal_map("cleo_inverse", inverse, 0, 1.1, 2048, 2048,
                "patterns/cleo.png");

  auto double_pole = [] __device__(Complex z) { return pow(z, -2); };
  conformal_map("double_pole", double_pole, 0, 1, 2048, 2048,
                "patterns/cannon.png");
  conformal_map("clock_double_pole", double_pole, 0, 2, 3200, 2400,
                "patterns/clock.png");

  auto two_cosh = [] __device__(Complex z) { return 2 * cosh(z); };
  conformal_map("two_cosh", two_cosh, 0, M_PI, 2048, 2048,
                "patterns/sirby.png");

  auto roots_of_unity = [] __device__(Complex z) { return pow(z, 3) - 1; };
  domain_color("roots_of_unity", roots_of_unity, 0, 2, 2048, 2048);

  // Prototypical example of a branch cut
  // mathworld.wolfram.com/BranchCut.html
  auto branch_cut = [] __device__(Complex z) { return sqrt(z); };
  domain_color("branch_cut", branch_cut, 0, 2, 2048, 2048);

  auto removable_singularity = [] __device__(Complex z) { return sin(z) / z; };
  domain_color("removable_singularity", removable_singularity, 0, 2 * M_PI,
               2048, 2048);

  auto triple_pole = [] __device__(Complex z) { return 1 / pow(z, 3); };
  domain_color("triple_pole", triple_pole, 0, 2, 2048, 2048);

  // Prototypical example of an essential singularity
  // mathworld.wolfram.com/EssentialSingularity.html
  auto essential_singularity = [] __device__(Complex z) { return exp(1 / z); };
  domain_color("essential_singularity", essential_singularity, 0, 0.5, 2048,
               2048);

  auto cluster_point = [] __device__(Complex z) { return 1 / tan(M_PI / z); };
  domain_color("cluster_point", cluster_point, 0, 1, 2048, 2048);

  auto sin_recip = [] __device__(Complex z) { return sin(1 / z); };
  domain_color("sin_recip", sin_recip, 0, 0.5, 2048, 2048);
  conformal_map("sin_recip", sin_recip, 0, 1, 2048, 2048,
                "patterns/checkerboard.png");

  auto spherify = [] __device__(Complex z) { return z * abs(z); };
  conformal_map("spherify", spherify, 0, 8, 2048, 2048,
                "patterns/checkerboard.png");

  auto whirlpool = [i] __device__(Complex z) {
    return 3 * z * exp(i * abs(z));
  };
  conformal_map("whirlpool", whirlpool, 0, 2, 2048, 2048,
                "patterns/checkerboard.png");

  auto knot = [] __device__(Complex z) { return 1 / sinh(z); };
  conformal_map("knot", knot, 0, 0.6, 2048, 2048, "patterns/checkerboard.png");

  auto multibrot = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = pow(z, 7) + z;
    }
    return z;
  };
  conformal_map("multibrot_flower", multibrot, 0, 1.4, 2048, 2048,
                "patterns/flower.png");
  conformal_map("multibrot", multibrot, 0, 1.5, 2048, 2048,
                "patterns/checkerboard.png");

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

  auto iterated_map1 = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = z * sin(pow(z, 3));
    }
    return z;
  };
  domain_color("iterated_map1", iterated_map1, 0, 2, 2048, 2048);

  auto iterated_map2 = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = sin(z) / pow(z, 2);
    }
    return z;
  };
  domain_color("iterated_map2", iterated_map2, 0, 0.15, 2048, 2048);

  auto iterated_map3 = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = z.real() * log(z) * exp(i * arg(z));
    }
    return z;
  };
  domain_color("iterated_map3", iterated_map3, 0, 2, 2048, 2048);

  auto iterated_map4 = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = tanh(1 / z) * exp(i * abs(z));
    }
    return z;
  };
  domain_color("iterated_map4", iterated_map4, 0, 2, 2048, 2048);

  auto iterated_map5 = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = z.real() * exp(i * z.imag());
    }
    return z;
  };
  domain_color("iterated_map5", iterated_map5, 0, 2, 2048, 2048);

  auto iterated_map6 = [] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = z / (pow(z, 3) - 1).real();
    }
    return z;
  };
  domain_color("iterated_map6", iterated_map6, 0, 4, 2048, 2048);

  auto iterated_map7 = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(i / pow(z, 2));
    }
    return z;
  };
  domain_color("iterated_map7", iterated_map7, 0, 3, 2048, 2048);

  auto iterated_map8 = [] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(1 / pow(z, 2));
    }
    return z;
  };
  domain_color("iterated_map8", iterated_map8, 0, 3, 2048, 2048);

  auto iterated_map9 = [] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(1 / pow(z, 3));
    }
    return z;
  };
  domain_color("iterated_map9", iterated_map9, 0, 3, 2048, 2048);

  auto iterated_map10 = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(z / (pow(z, 3) - i));
    }
    return z;
  };
  domain_color("iterated_map10", iterated_map10, 0, 3, 2048, 2048);

  auto iterated_map11 = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = log(pow(1 / z, z));
    }
    return z;
  };
  domain_color("iterated_map11", iterated_map11, 0, 2, 2048, 2048);

  auto iterated_map12 = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = exp(i * z.imag()) * tan(1 / z);
    }
    return z;
  };
  domain_color("iterated_map12", iterated_map12, 0, 2, 2048, 2048);

  auto iterated_map13 = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = exp(i * abs(z)) * exp(1 / z);
    }
    return z;
  };
  domain_color("iterated_map13", iterated_map13, 0, 2, 2048, 2048);

  auto needle = [] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = z.imag() * log(z);
    }
    return z;
  };
  domain_color("needle", needle, 0, 2, 2048, 2048);

  auto tron = [i] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = z / abs(pow(z, 3) - 1);
    }
    return z * exp(i * arg(sin(pow(z, 3) - 1)));
  };
  domain_color("tron", tron, 0, 1.2, 2048, 2048);

  auto grace22 = [] __device__(Complex z) {
    Complex w = z;
    for (int n = 0; n < 22; n++) {
      w = arg(w) * exp(w);
    }
    return w;
  };
  domain_color("grace22", grace22, M_PI_2 * i, 0.4, 2048, 2048);

  auto pollock = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(((1 - i) * pow(z, 6) + (7 + i) * z) / (2 * pow(z, 5) + 6));
    }
    return z;
  };
  domain_color("pollock", pollock, 0, 2, 2048, 2048);

  auto i_of_storm = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(((1 - i) * pow(z, 4) + (7 + i) * z) / (2 * pow(z, 5) + 6));
    }
    return z;
  };
  domain_color("i_of_storm", i_of_storm, 0, 2, 2048, 2048);

  auto pink_and_green_slime = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = tan(1 / z) * exp(i * abs(z));
    }
    return z;
  };
  domain_color("pink_and_green_slime", pink_and_green_slime, 0, 2.5, 2048,
               2048);

  auto kaleidoscope = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = cos(1 / z);
    }
    return log(z);
  };
  domain_color("kaleidoscope", kaleidoscope, 0, 2, 2048, 2048);

  auto strawberry_banana = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log((z + i) / pow(z, 2));
    }
    return z;
  };
  domain_color("strawberry_banana", strawberry_banana, 0, 3, 2048, 2048);

  auto triangle_dragon = [i] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = log(((1 - i) * pow(z, 6) - 4 * i * pow(z, 3)) / (2 * pow(z, 6)));
    }
    return z;
  };
  domain_color("triangle_dragon", triangle_dragon, -0.3 * i, 3, 2048, 2048);

  auto nightshade = [i] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
      z = exp(z * z + i);
    }
    return z;
  };
  domain_color("nightshade", nightshade, 0, 3, 2048, 2048);

  auto glitch = [] __device__(Complex z) {
    for (int n = 0; n < 64; n++) {
      z = exp(1 / log(z));
    }
    return z;
  };
  domain_color("glitch", glitch, 0, 2, 2048, 2048);

  auto acid_pitchfork = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = exp(i * arg(z)) * tanh(1 / pow(sinh(1 / z), 2));
    }
    return z;
  };
  domain_color("acid_pitchfork", acid_pitchfork, 0, 2, 2048, 2048);

  auto carpet = [] __device__(Complex z) {
    for (int n = 0; n < 16; n++) {
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

  auto pastel_oology = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = exp(i * abs(z)) * log(z);
    }
    return z;
  };
  domain_color("pastel_oology", pastel_oology, 0, 2, 2048, 2048);

  auto something_in_my_skin = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = exp(i * arg(z * z)) * (z * z - i);
    }
    return z;
  };
  domain_color("something_in_my_skin", something_in_my_skin, 0, 2, 2048, 2048);

  auto ex_nihilo = [i] __device__(Complex z) {
    for (int n = 0; n < 8; n++) {
      z = exp(i * (z * z).imag()) * (z * z + i);
    }
    return z;
  };
  domain_color("ex_nihilo", ex_nihilo, 0, 2, 2048, 2048);

  auto under_construction = [] __device__(Complex z) {
    for (int n = 0; n < 32; n++) {
      z = sin(1 / z);
    }
    return log(z);
  };
  domain_color("under_construction", under_construction, 0, 2, 2048, 2048);

  return 0;
}