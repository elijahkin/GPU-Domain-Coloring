def mobius(z):
  for _ in range(3):
    z = np.log((z**13-1) / (2*z+1j))
  return z
domain_color(mobius, 0, 2, 2048)

def mobius2(z):
  for _ in range(3):
    z = np.log(1/(z**4-1))
  return z
domain_color(mobius2, 0, 2, 2048)

def sky(x):
  for _ in range(16):
    x = np.log( (x**2+x+1) / (3*x**2+2*x+1) )
  return x
domain_color(sky, 0, 4, 2048)

def accum9(z):
  for _ in range(64):
    # z = np.log(z)
    z = z**2-z-1
    z = 1/(z**2+7)
    #* np.exp(1j*np.absolute(1/(z**2+1)))
    # z = (z-1j)**(1/2)
  return z
domain_color(accum9, 0, 8, 2048)

def test(z):
  return z**2-z-1
domain_color(test, 0, 2, 2048)

def poly(z):
  for _ in tqdm(range(32)):
    z = np.log((z-1)/(z**2+1))
  return z

def banana_poly5(z):
  for _ in tqdm(range(32)):
    z = np.log(1j / z**3)
  return z

def banana_poly6(z):
  for _ in tqdm(range(32)):
    z = np.log((z**5+1j) / (z**6))
  return z

def banana_poly9(z):
  for _ in tqdm(range(32)):
    z = np.log(((1+1j)*z**3 - 1)/(2 * z**3))
  return z