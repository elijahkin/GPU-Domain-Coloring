from complex_plot import *

def escape_time(fractal, iterations, center, r, quality, cmap):
    c = complex_region(center, r, quality)
    z = np.zeros(c.shape)
    color = np.zeros(c.shape)
    for _ in tqdm(range(iterations)):
        # Apply the iterated map...
        z = fractal(z, c)
        # ...and color the points that escaped this iteration
        escaped = np.where(np.absolute(z) > 2)
        z[escaped] = 0
        c[escaped] = 0
        color[escaped] = (_ / iterations) * 256
    complex_plot(color, f'renders/escape time of {fractal.__name__} ({center}, {r}, {iterations}, {quality}).png', cmap)

def mandelbrot(z, c):
    return z**2 + c

def burning_ship(z, c):
    return (np.abs(z.real) + 1j*np.abs(z.imag))**2 + c

def multibrot(z, c):
    return z**7 + c

def tricorn(z, c):
    return np.conjugate(z)**2 + c

escape_time(mandelbrot, 200, -0.765, 1.435, 256, matplotlib.cm.cubehelix)
# escape_time(mandelbrot, 500, -0.77568377+0.13646737j, 0.0000001, 4096, matplotlib.cm.cubehelix) # Misiurewicz point M_23,2
# escape_time(mandelbrot, 2000, 0.001643721971153-0.822467633298876j, 0.00000000002, 2048, matplotlib.cm.cubehelix)
# escape_time(burning_ship, 50, 0, 2.2, 2048, matplotlib.cm.cubehelix)
# escape_time(multibrot, 10, 0, 1.4, 8192, matplotlib.cm.Blues) # Multibrot snowflake with d=7 made for Grace for Christmas 2022
# escape_time(tricorn, 200, 0, 2.2, 4096, matplotlib.cm.cubehelix)
# escape_time(mandelbrot, 150, -0.765, 1.435, 4096, matplotlib.cm.YlGn) # Mandelbrot à la chartreuse for Caleb following our MATH 4652 final

# def newton1(z, c):
#     return z - (z**3-1)/(3*z**2)
# escape_time(newton1, 32, 0, 2, 2048, matplotlib.cm.hsv)