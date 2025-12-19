# simpix

C++ starter code
* simpix_start.cpp
use make to build this example

Usage: simapix_start image1 image2 <output=out.png>

Python starter code
* simpix_start.py

Usage: simapix_start image1 image2 <output=out.png>

# simpix — Pixel Mapping via Simulated Annealing

This repo implements the pixel-mapping (pixel permutation) algorithm discussed in class using simulated annealing.  
Given a **source** image A and a **target** image B (same dimensions), the algorithm searches for a permutation of A’s pixels that minimizes the total squared RGB error relative to B.

---

## Requirements

Python 3 with:
- `numpy`
- `Pillow`
- `matplotlib`

---

## How to run

Images must have the same size. For PHYS3630, the minimum is **640×480**.

### Run A → B (IMG_1738 → IMG_1884)

```bash
python simpix1.py IMG_1738_640x480.png IMG_1884_640x480.png \
  1738_to_1884_640x480.png 50000 800 0.97 1e-2 1

