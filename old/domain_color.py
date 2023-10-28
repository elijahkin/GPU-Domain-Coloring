from complex_plot import *
from matplotlib.colors import hsv_to_rgb
import warnings
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Hide warnings produced by overflow errors
warnings.filterwarnings('ignore')

def complex_to_rgb(z):
  H = (np.angle(z) + np.pi) / (2 * np.pi)
  S = np.ones(H.shape)
  L = 2 / np.pi * np.arctan(np.abs(z))

  HSL = np.stack((H, S, L), axis=2)
  return hsv_to_rgb(HSL)

def domain_color(f, center, r, quality):
  z = complex_region(center, r, quality)
  w = f(z)
  rgb = complex_to_rgb(w)
  complex_plot(rgb, f'renders/domain color of {f.__name__} ({center}, {r}, {quality}).png', matplotlib.cm.hsv)

#   fig, ax = plt.subplots(figsize=(12, 12))
#   divider = make_axes_locatable(ax)
#   cax = divider.append_axes("right", size="5%", pad=0.2)
#   cax2 = divider.append_axes("right", size="5%", pad=0.6)

#   bar_data = ax.imshow(np.array([[(0, 0, 0)]]), vmin=0, vmax=255, cmap='gray')
#   cbar2 = plt.colorbar(bar_data, cax2, ticks=[0, 255])
#   cbar2.ax.set_yticklabels(['0', '∞'])
#   map = ax.imshow(rgb, cmap='hsv', vmin=0, vmax=255, interpolation='none', extent=[x_min, x_max, y_min, y_max])
#   cbar = plt.colorbar(map, cax, ticks=[0, 64, 128, 192, 255])
#   cbar.ax.set_yticklabels(['- π', '- π / 2', '0', 'π / 2', 'π'])

#   ax.set_xlabel('Re(z)')
#   ax.set_ylabel('Im(z)')
#   # plt.show()
#   plt.savefig(f'lines.png', dpi=256, bbox_inches='tight', pad_inches=0)

def identity(z):
  return z

def roots_of_unity(z):
  return z**3 - 1

# Prototypical example of a branch cut
# https://mathworld.wolfram.com/BranchCut.html
def sqrt(z):
  return z**0.5

def removable_singularity(z):
  return np.sin(z) / z

def pole_of_order_3(z):
  return z**-3

# Prototypical example of an essential singularity
# https://mathworld.wolfram.com/EssentialSingularity.html
def essential_singularity(z):
  return np.exp(1/z)

def cluster_point(z):
  return np.tan(1/z)

def riemann_zeta(z):
  w = 0
  for n in tqdm(range(1, 32)):
    w += n**-z
  return w

# Prototypical example of a natural boundary
# https://mathworld.wolfram.com/NaturalBoundary.html
def lacunary(z):
  w = 0
  for _ in tqdm(range(256)):
    w += z
    z = z**2
  return w

def tetration(z):
  w = z
  for _ in tqdm(range(31)):
    w = z**w
  return w

def iterated_carpet(z):
  for _ in tqdm(range(256)):
    z = np.cos(z) / np.sin(z**4 - 1)
  return z

def iterated_crevice(z):
  for _ in tqdm(range(128)):
    z = np.sin(1/z)
  return z

def iterated_spikes(z):
  for _ in tqdm(range(128)):
    z = 1/z**2 * np.sin(z)
  return z

def iterated_map(z):
  for _ in tqdm(range(128)):
    z = z*np.sin(z**3)
  return z

# https://people.math.sc.edu/howard/Sisson/SissonPaper.pdf
def i_of_medusa(z):
  for _ in tqdm(range(32)):
    z = np.log(((1-1j)*z**6+(7+1j)*z)/(2*z**5+6))
  return z

def i_of_storm(z):
  for _ in tqdm(range(32)):
    z = np.log(((1-1j)*z**4+(7+1j)*z)/(2*z**5+6))
  return z

def poly(z):
  for _ in tqdm(range(32)):
    z = np.log((z-1)/(z**2+1))
  return z

# Newton fractal for z^3-1
# https://en.wikipedia.org/wiki/Newton_fractal
def newton1(z):
  for _ in tqdm(range(32)):
    z -= (z**3-1)/(3*z**2)
  return z

def newton2(z):
  for _ in tqdm(range(32)):
    z -= (z**3+1)/(3*z**2)
  return z

def newton3(z):
  for _ in tqdm(range(50)):
    z -= (z**8+15*z**4-16)/(8*z**7+60*z**3)
  return z

def newton4(z):
  for _ in tqdm(range(50)):
    z -= (z**6+z**3-1)/(6*z**5+3*z**2)
  return z

def newton5(z):
  for _ in tqdm(range(100)):
    z -= (z**32-1)/(32*z**31)
  return z

def newton6(z):
  for _ in tqdm(range(50)):
    z -= (z**4+z**3+z**2+z+1)/(4*z**3+3*z**2+2*z+1)
  return z

def sinarc(z):
  return np.sin(1/z)

def cluster_point2(z):
  return 1/np.sin(np.pi/z)

def banana_poly(z):
  for _ in tqdm(range(32)):
    z = np.log((z+1j) / (z**2))
  return z

def banana_poly2(z):
  for _ in tqdm(range(32)):
    z = np.log(1j / z**2)
  return z

def banana_poly3(z):
  for _ in tqdm(range(32)):
    z = np.log(1 / z**2)
  return z

def banana_poly4(z):
  for _ in tqdm(range(32)):
    z = np.log(1 / z**3)
  return z

def banana_poly5(z):
  for _ in tqdm(range(32)):
    z = np.log(1j / z**3)
  return z

def banana_poly6(z):
  for _ in tqdm(range(32)):
    z = np.log((z**5+1j) / (z**6))
  return z

def banana_poly7(z):
  for _ in tqdm(range(32)):
    z = np.log((z) / (z**3-1j))
  return z

def banana_poly8(z):
  for _ in tqdm(range(32)):
    z = np.log(((1-1j)*z**6 - 4j*z**3)/(2 * z**6))
  return z

def banana_poly9(z):
  for _ in tqdm(range(32)):
    z = np.log(((1+1j)*z**3 - 1)/(2 * z**3))
  return z

# domain_color(identity, 0, 1, 2048)
# domain_color(roots_of_unity, 0, 2, 2048)
# domain_color(sqrt, 0, 2, 2048)
# domain_color(removable_singularity, 0, np.pi, 2048)
# domain_color(pole_of_order_3, 0, 3, 2048)
# domain_color(essential_singularity, 0, 1/2, 2048)
# domain_color(cluster_point, 0, 1, 2048)
# domain_color(riemann_zeta, 0, 7, 2048)
# domain_color(lacunary, 0, 1, 2048)
# domain_color(tetration, 0, 3, 2048)
# domain_color(iterated_carpet, 0, 2, 2048)
# domain_color(iterated_crevice, 0, 1.07, 2048)
# domain_color(iterated_spikes, 0, .15, 2048)
# domain_color(iterated_map, 0, 2, 2048)
# domain_color(i_of_medusa, 0, 2, 1024)
# domain_color(i_of_storm, 0, 2, 2048)
# domain_color(newton6, 0, 2, 2048)
# domain_color(sinarc, 0, 0.5, 2048)
# domain_color(cluster_point2, 0, 1.5, 2048)

# domain_color(banana_poly, 0, 3, 4096)
# domain_color(banana_poly2, 0, 3, 4096)
# domain_color(banana_poly3, 0, 3, 2048)
# domain_color(banana_poly4, 0, 3, 4096)
# domain_color(banana_poly7, 0, 3, 2048)
# domain_color(banana_poly8, -0.3j, 3, 4096)

def topological_self_portrait(z):
  # compute primitive root of unity
  w = np.exp(2j*np.pi / 7)
  # compute distance to nearest root of unity
  d = np.abs(z - 1)
  for p in range(1, 7):
    d = np.minimum(d, np.abs(z - w**p))
  # if z at least 2 from origin, color it the same color as roots of unity
  d[np.abs(z) > 2] = 0
  # return whether distance is less than some radius
  # d[d < 0.3] = z
  z[d > 0.3] = 0
  z[d <= 0.3] = 2*np.exp(3*np.pi*1j/4)
  return z

# domain_color(topological_self_portrait, 0, 2.5, 4096)

def quilt(z):
  slice = np.angle(z) // (2*np.pi / 12)
  w = slice

  w[slice == 0] = 0
  w[slice == 5] = 0

  w[slice == -1] = 0
  w[slice == -6] = 0

  w[slice == 1] = np.exp(-3*np.pi*1j/4)
  w[slice == 2] = np.exp(-3*np.pi*1j/4)

  w[slice == 3] = np.exp(-1*np.pi*1j/4)
  w[slice == 4] = np.exp(-1*np.pi*1j/4)

  w[slice == -2] = np.exp(1*np.pi*1j/4)
  w[slice == -3] = np.exp(1*np.pi*1j/4)

  w[slice == -4] = np.exp(3*np.pi*1j/4)
  w[slice == -5] = np.exp(3*np.pi*1j/4)

  return w

# domain_color(quilt, 0, 1, 2048)

def quilt2(z):

  # z_new = (z.real % 1) + 1j*(z.imag % 1)
  # z = z_new

  slice = (np.angle(np.exp(np.pi*1j/6)*z)) // (2*np.pi / 6)
  w = slice

  w[slice == -1] = 0
  w[slice == 0] = 0

  w[slice == 1] = 1
  w[slice == 2] = 1

  w[slice == -2] = -1
  w[slice == -3] = -1

  return w

# domain_color(quilt2, 0, 3, 2048)

def mobius(z):
  for _ in range(3):
    z = np.log((z**13-1) / (2*z+1j))
  return z

def mobius2(z):
  for _ in range(3):
    z = np.log(1/(z**4-1))
  return z

# domain_color(mobius, 0, 2, 2048)
# domain_color(mobius2, 0, 2, 2048)

def sky(x):
  for _ in range(16):
    x = np.log( (x**2+x+1) / (3*x**2+2*x+1) )
  return x

# domain_color(sky, 0, 4, 2048)

def kin2(z):
  for _ in range(32):
    z = np.log(z**(1/z))
  return z

# domain_color(kin2, 0, 32, 2048)

def kin3(z):
  for _ in range(16):
    z = np.log((1/z)**z)
  return z

# domain_color(kin3, 0, 2, 2048)

def kin4(z):
  for _ in range(16):
    z = np.exp(z**2+1j)
  return z

# domain_color(kin4, 0, 3, 2048)

def kin5(z):
  for _ in range(64):
    z = np.exp(1/np.log(z))
  return z

# domain_color(kin5, 0, 2, 2048)

def accum(z):
  for _ in range(8):
    z = np.exp(1j*np.angle(z))*np.tanh(1/np.sinh(1/z)**2)
  return z

# domain_color(accum, 0, 2, 1920)

def accum2(z):
  for _ in range(8):
    z = np.exp(1j*np.absolute(z))*np.tan(1/z)
  return z

# domain_color(accum2, 0, 2, 2048)

def accum3(z):
  for _ in range(8):
    z = np.exp(1j*np.imag(z))*np.tan(1/z)
  return z

# domain_color(accum3, 0, 2, 2048)

def twisty_exp(z):
  for _ in range(8):
    z = np.exp(1j*np.absolute(z))*np.exp(1/z)
  return z

# domain_color(twisty_exp, 0, 2, 2048)

def log_egg(z):
  for _ in range(8):
    z = np.exp(1j*np.absolute(z))*np.log(z)
  return z

# domain_color(log_egg, 0, 2, 2048)

def something_in_my_skin(z):
  for _ in range(8):
    z = np.exp(1j*np.angle(z**2))*(z**2-1j)
  return z

# domain_color(something_in_my_skin, 0, 2, 2048)

def ex_nihilo(z):
  for _ in range(8):
    z = np.exp(1j*np.imag(z**2))*(z**2+1j)
  return z

# domain_color(ex_nihilo, 0, 2, 2048)

def i_want_to_go_home_now(z):
  for _ in range(16):
    z = np.cos(1/z)
  return np.log(z)

# domain_color(i_want_to_go_home_now, 0, 2, 2048)

def under_construction(z):
  for _ in range(32):
    z = np.sin(1/z)
  return np.log(z)

# domain_color(under_construction, 0, 2, 2048)

def accum7(z):
  for _ in range(16):
    z = np.tan(1/z)*np.exp(1j*np.absolute(z))
  return z

# domain_color(accum7, 0, 2.5, 2048)

def accum8(z):
  for _ in range(16):
    z = np.arctan(1/(z**2+1))*np.exp(1j*np.real(z))
  return z

# domain_color(accum8, 0, 3, 2048)

def accum9(z):
  for _ in range(64):
    # z = np.log(z)
    z = z**2-z-1
    z = 1/(z**2+7)
    #* np.exp(1j*np.absolute(1/(z**2+1)))
    # z = (z-1j)**(1/2)
  return z

# domain_color(accum9, 0, 8, 2048)

def test(z):
  return z**2-z-1

# domain_color(test, 0, 2, 2048)

def up_and_coming(z):
  for _ in range(8):
    z = np.imag(z) * np.log(z)
  return z

# domain_color(up_and_coming, 0, 2, 1024)

def up_and_coming2(z):
  for _ in range(8):
    z = np.real(z) * np.log(z) * np.exp(1j * np.angle(z))
  return z

# domain_color(up_and_coming2, 0, 2, 1024)

def up_and_coming3(z):
  for _ in range(64):
    z = np.tanh(1/z) * np.exp(1j * np.absolute(z))
  return z

# domain_color(up_and_coming3, 0, 2, 1024)

def up_and_coming4(z):
  for _ in range(64):
    z = np.real(z) * np.exp(1j * np.imag(z))
  return z

# domain_color(up_and_coming4, 0, 2, 1024)

def up_and_coming5(z):
  for _ in range(64):
    z = z / np.absolute(z**3 - 1)
  return z * np.exp(1j * np.angle(np.sin(z**3-1)))

domain_color(up_and_coming5, 0, 1.2, 1024)

def up_and_coming6(z):
  for _ in range(64):
    z = z / np.real(z**3 - 1)
  return z

# domain_color(up_and_coming6, 0, 4, 1024)

def up_and_coming7(z):
  for _ in range(64):
    z = z**((z**2 + 1j)**(1/2))
  return z

# domain_color(up_and_coming7, 0, 2, 1024)