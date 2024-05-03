## Overview
This repository contains code written for my thesis project at the University of Virginia, for which I am developing a library for plotting fractals and functions of a complex variable using NVIDIA graphics cards.

For reading and writing images, we use the `.ppm` format, examples of which can be found [here](https://en.wikipedia.org/wiki/Netpbm). We chose this format because it is simple enough to make a reader/writer from scratch, and can be easily converted to more standard formats such as `.jpg` and `.png` via ImageMagick. The downside of this format is that file sizes tend to be large.

## Domain Coloring
Given a complex number $z$, the function we use produces a color in the $HSL$ color model given by
```math
\begin{align*} H &= \frac{\arg z}{2\pi} + \frac{1}{2} \\ S &= 1 \\ L &= \frac{2}{\pi}\arctan|z| \end{align*}
```
which we then convert to the $RGB$ model using the procedure detailed by
[Saravanan, Yamuna, and Nandhini](https://ieeexplore.ieee.org/abstract/document/7754179).

### Phase Portraits
Phase portraiture is a technique much like domain coloring, except that

## Conformal Mapping
[Mercat](https://en.wikibooks.org/wiki/Fractals/Conformal_map)

## Escape-Time Fractals

### Normalization
In order to avoid color banding,

## To-Do List
* Finish copying over examples and delete Python versions
* Unify domain color and conformal map code as much as possible, update file structure
* Command line progress bar?
* Write CPU version and compare performance
* Phase portraits
* Reading from .csv file on website
* Update renders on website and write descriptions
* Long term: OpenGL

## Resources

* https://fredrikj.net/blog/2022/02/computing-the-lerch-transcendent/

* https://users.mai.liu.se/hanlu09/complex/
* http://www.javaview.de/domainColoring/

* https://math.okstate.edu/people/scurry/5283/sp21/17_Wegert2016_Chapter_VisualExplorationOfComplexFunctions.pdf
* https://www.ams.org/notices/201106/rtx110600768p.pdf

* https://apps.dtic.mil/sti/tr/pdf/ADA210016.pdf

### Window
* `sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev`
* `nvcc -o bin/window src/window.cu -lGL -lglut --extended-lambda`
