from complex_plot import *
from PIL import Image
import warnings

# Hide warnings produced by overflow errors
warnings.filterwarnings('ignore')

def conformal(f, image, center, r, quality):
    img = np.asarray(Image.open(image))
    w_mat = np.zeros((quality, quality, 3))

    z = complex_region(center, r, quality)
    w = f(z)
    w = np.clip(w, -10**12, 10**12)

    # get the real and imaginary parts of the resulting complex numbers
    u, v = w.real, w.imag
    # map the real and imaginary parts to the pixel coordinates of the image
    p = ((u - (center.real - r)) / (2*r)) * img.shape[0]
    q = ((v - (center.imag - r)) / (2*r)) * img.shape[1]
    # wrap around the indices that are out of bounds
    p = p % img.shape[0]
    q = q % img.shape[1]
    # create the output image by indexing into the input image

    p = np.nan_to_num(p)
    q = np.nan_to_num(q)

    w_mat = img[p.astype(int), q.astype(int)]

    complex_plot(w_mat, f'renders/conformal map of {f.__name__} ({center}, {r}, {image[9:-4]}, {quality}).png', None)

def inverse(z):
    return 1/z

def two_cosh(z):
    return 2*np.cosh(z)

def sinarc(z):
    return np.sin(1/z)

def f(z):
    return 1/np.cos(z)

def square(z):
    return z**2

def identity(z):
    return z

def shear(z):
    return 2*z * np.exp(1j * np.pi / 4)

def multibrot(z):
    for _ in tqdm(range(16)):
        z = z**7 + z
    return z

def spherify(z):
    return z*np.absolute(z)

def whirlpool(z):
    return 3*z*np.exp(1j * np.absolute(z))

def double_pole(z):
    return z**-2

def knot(z):
    return 1/np.sinh(z)

def knot2(z):
    for _ in tqdm(range(3)):
        z = np.sin(1.5*z)
    return z

conformal(inverse, 'patterns/longeli.jpg', 0, 1, 2048)
conformal(two_cosh, 'patterns/sirby.jpg', 0, np.pi, 2048)
conformal(inverse, 'patterns/cleo.jpg', 0, 1.1, 2048)
conformal(double_pole, 'patterns/cannon.jpg', 0, 1, 2048)
conformal(multibrot, 'patterns/flower.jpg', 0, 1.4, 2048)
conformal(sinarc, 'patterns/domino.png', 0, 1, 2048)
conformal(shear, 'patterns/domino.png', 0, 1, 2048)
conformal(multibrot, 'patterns/domino.png', 0, 1.5, 4096)
conformal(spherify, 'patterns/domino.png', 0, 8, 2048)
conformal(whirlpool, 'patterns/domino.png', 0, 2, 2048)
conformal(knot, 'patterns/web.jpg', 0, 0.6, 2048)
conformal(inverse, 'patterns/blossom.jpg', 0, np.pi/8, 2048)
conformal(knot2, 'patterns/web.jpg', 0, 1, 2048)