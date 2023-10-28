#include <complex>
#include <iostream>
#include <string>
#include <valarray>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

valarray<complex<double>> complex_region(complex<double> center, double radius, int N) {
    double step_size = 2 * radius / (N - 1);
    valarray<complex<double>> region (N * N);
    double re, im;

    im = center.imag() + radius;
    for (int row = 0; row < N; row++) {
        re = center.real() - radius;
        for (int col = 0; col < N; col++) {
            region[row * N + col] = re + im * 1i;
            re += step_size;
        }
        im -= step_size;
    }
    return region;
}

unsigned char * z_to_rgb(valarray<complex<double>> z) {
    int n_pixels = z.size();
    static unsigned char * rgb;
    rgb = (unsigned char *) malloc(n_pixels * 3);

    for (int i = 0; i < n_pixels; i++) {
        // HSL scheme adapted from https://en.wikipedia.org/wiki/Domain_coloring
        double hue = (arg(z[i]) / (2 * M_PI)) + 0.5;
        double saturation = 1.0;
        double lightness = (2 / M_PI) * atan(abs(z[i]));

        double chroma = (1 - std::abs(2 * lightness - 1)) * saturation;
        double x = chroma * (1 - std::abs(fmod(6 * hue, 2) - 1));

        double r, g, b;
        if (hue >= 0 && hue < 1.0/6.0) {
            r = chroma;
            g = x;
            b = 0;
        } else if (hue >= 1.0/6.0 && hue < 2.0/6.0) {
            r = x;
            g = chroma;
            b = 0;
        } else if (hue >= 2.0/6.0 && hue < 3.0/6.0) {
            r = 0;
            g = chroma;
            b = x;
        } else if (hue >= 3.0/6.0 && hue < 4.0/6.0) {
            r = 0;
            g = x;
            b = chroma;
        } else if (hue >= 4.0/6.0 && hue < 5.0/6.0) {
            r = x;
            g = 0;
            b = chroma;
        } else {
            r = chroma;
            g = 0;
            b = x;
        }

        double m = lightness - 0.5 * chroma;
        r += m;
        g += m;
        b += m;

        rgb[3*i]   = static_cast<int>(r * 255);
        rgb[3*i+1] = static_cast<int>(g * 255);
        rgb[3*i+2] = static_cast<int>(b * 255);
    }
    return rgb;
}

unsigned char * z_to_rgb2(valarray<complex<double>> z) {
    int n_pixels = z.size();
    static unsigned char * rgb;
    rgb = (unsigned char *) malloc(n_pixels * 3);

   uint8_t sextant;
    double h, s, l;
    double chroma, x, m;
    double r, g, b;

    for (int i = 0; i < n_pixels; i++) {
        h = (arg(z[i]) / (2 * M_PI)) + 0.5;
        s = 1;
        l = (2 / M_PI) * atan(abs(z[i]));

        chroma = (1 - abs(2 * l - 1)) * s;
        x = chroma * (1 - abs(fmod(6 * h, 2) - 1));
        m = l - (chroma / 2);

        // http://0x80.pl/notesen/2019-02-03-simd-switch-implementation.html
        sextant = int(h * 6);
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

        rgb[3 * i + 0] = (uint8_t) ((r + m) * 255);
        rgb[3 * i + 1] = (uint8_t) ((g + m) * 255);
        rgb[3 * i + 2] = (uint8_t) ((b + m) * 255);
    }
    return rgb;
}

void domain_color(valarray<complex<double>> (*f)(valarray<complex<double>>),
                  complex<double> center, double radius, int N, string filename) {
    valarray<complex<double>> z = complex_region(center, radius, N);
    z = f(z);

    // Fix NaN values that may cause weird highlighting in the visuals
    for (int i = 0; i < z.size(); i++) {
        if (isnan(z[i].real()) || isnan(z[i].imag())) {
            z[i] = 0;
        }
    }

    unsigned char *p = z_to_rgb2(z);
    filename = "renders/" + filename + " " + to_string(N) + ".png";
    stbi_write_png(&filename[0], N, N, 3, p, 0);
}

valarray<complex<double>> identity(valarray<complex<double>> z) {
    return z;
}

valarray<complex<double>> roots_of_unity(valarray<complex<double>> z) {
    return pow(z,3)-1;
}

valarray<complex<double>> square_root(valarray<complex<double>> z) {
    return sqrt(z);
}

valarray<complex<double>> pole_of_order_3(valarray<complex<double>> z) {
    return 1/pow(z, 3);
}

valarray<complex<double>> removable_singularity(valarray<complex<double>> z) {
    return sin(z)/z;
}

valarray<complex<double>> essential_singularity(valarray<complex<double>> z) {
    return exp(1/z);
}

valarray<complex<double>> cluster_point(valarray<complex<double>> z) {
    return 1 / tan(M_PI / z);
}

valarray<complex<double>> lacunary(valarray<complex<double>> z) {
    valarray<complex<double>> w (z.size());
    w = 0;
    for (int i = 0; i < 64; i++) {
        w += z;
        z *= z;
    }
    return w;
}

valarray<complex<double>> tetration(valarray<complex<double>> z) {
    valarray<complex<double>> w (z.size());
    w = z;
    for (int i = 0; i < 31; i++) {
        w = pow(z, w);
    }
    return w;
}

valarray<complex<double>> newton1(valarray<complex<double>> z) {
    for (int i = 0; i < 8; i++) {
        z -= (pow(z, 3) - 1) / (3*pow(z, 2));
    }
    return z;
}

valarray<complex<double>> newton2(valarray<complex<double>> z) {
    for (int i = 0; i < 8; i++) {
        z -= (pow(z, 3) + 1) / (3*pow(z, 2));
    }
    return z;
}

valarray<complex<double>> newton3(valarray<complex<double>> z) {
    for (int i = 0; i < 8; i++) {
        z -= (pow(z, 8) + 15*pow(z, 4) - 16) / (8*pow(z, 7) + 60*pow(z, 3));
    }
    return z;
}

valarray<complex<double>> newton4(valarray<complex<double>> z) {
    for (int i = 0; i < 8; i++) {
        z -= (pow(z, 6) + pow(z, 3) - 1) / (6*pow(z, 5) + 3*pow(z, 2));
    }
    return z;
}

valarray<complex<double>> newton5(valarray<complex<double>> z) {
    for (int i = 0; i < 8; i++) {
        z -= (pow(z, 32) - 1) / (32*pow(z, 31));
    }
    return z;
}

valarray<complex<double>> newton6(valarray<complex<double>> z) {
    for (int i = 0; i < 8; i++) {
        z -= (pow(z, 4) + pow(z, 3) + pow(z, 2) + z + 1) / (4*pow(z, 3) + 3*pow(z, 2) + 2*z + 1);
    }
    return z;
}

valarray<complex<double>> purple_and_green_slime(valarray<complex<double>> z) {
    for (int i = 0; i < 32; i++) {
        z = log((pow(z,6) - 1i*pow(z,6) + 7*z + 1i*z)/(2*pow(z,5)+6));
    }
    return z;
}

valarray<complex<double>> pink_and_green_slime(valarray<complex<double>> z) {
    for (int i = 0; i < 32; i++) {
        z = tan(1/z)*exp(1i*abs(z));
    }
    return z;
}

valarray<complex<double>> kaleidoscope(valarray<complex<double>> z) {
    for (int i = 0; i < 16; i++) {
        z = cos(1/z);
    }
    return log(z);
}

valarray<complex<double>> carpet(valarray<complex<double>> z) {
    for (int i = 0; i < 32; i++) {
        z = cos(z) / sin(pow(z, 4) - 1);
    }
    return z;
}

valarray<complex<double>> crevice(valarray<complex<double>> z) {
    for (int i = 0; i < 32; i++) {
        z = sin(1 / z);
    }
    return z;
}

int main() {
    auto start = chrono::steady_clock::now();
    // Complex Analysis Functions
    domain_color(&identity, 0, 1, 2048, "identity");
    // domain_color(&roots_of_unity, 0, 2, 2048, "roots of unity");
    // domain_color(&square_root, 0, 1, 2048, "square_root");
    // domain_color(&pole_of_order_3, 0, 2, 2048, "pole of order 3");
    // domain_color(&removable_singularity, 0, 4, 2048, "removable singularity");
    // domain_color(&essential_singularity, 0, 0.5, 4096, "essential singularity");
    // domain_color(&cluster_point, 0, M_PI, 2048, "cluster point");
    // domain_color(&lacunary, 0, 1, 2048, "lacunary");
    // domain_color(&tetration, 0, 3, 2048, "tetration");

    // Newton Fractals
    // domain_color(&newton1, 0, 2, 2048, "newton1");
    // domain_color(&newton2, 0, 2, 2048, "newton2");
    // domain_color(&newton3, 0, 2, 2048, "newton3");
    // domain_color(&newton4, 0, 2, 2048, "newton4");
    // domain_color(&newton5, 0, 2, 2048, "newton5");
    // domain_color(&newton6, 0, 2, 2048, "newton6");

    // Iterated maps
    // domain_color(&purple_and_green_slime, 0, 2, 2048, "purple and green slime");
    // domain_color(&pink_and_green_slime, 0, 2.5, 1024, "pink and green slime 2");
    // domain_color(&kaleidoscope, 0, 2, 2048, "kaleidoscope");
    // domain_color(&carpet, 0, 2, 2048, "carpet");
    // domain_color(&crevice, 0, 1.07, 2048, "crevice");
    auto end = chrono::steady_clock::now();
    cout << chrono::duration <double> (end - start).count() << endl;
    return 0;
}