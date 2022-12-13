# Project 5

## Dependencies

### C++
- [Armadillo](http://arma.sourceforge.net/) for numerical linear algebra.

### Python
- [NumPy](https://pypi.org/project/numpy/) for numerical linear algebra.
- [PyArmadillo](https://pyarma.sourceforge.io/) for numerical linear algebra and data handling.
- [Matplotlib](https://pypi.org/project/matplotlib/) for plotting.

### Build
- [Make](https://en.wikipedia.org/wiki/Make_(software)) for building.

## Build

**Note.** This code has only been tested on Ubuntu Linux. 

To compile, link and run the C++ code, use the command

```
$ make all
```

To generate all the plots, run

```
$ python3 generate_plots.py
```

To generate animations of the time-evolution of the probability density for the different potentials, run

```
$ python3 generate_animations.py
```
