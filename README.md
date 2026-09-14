# EN3160 Assignment 1 — Intensity Transformations and Neighborhood Filtering

**Gunarathna K. T. M. B. — 230203G**

Solutions to all 10 questions of EN3160 Assignment 1 (Ranga Rodrigo): piecewise-linear
intensity transforms, gray/white-matter accentuation, gamma correction in L\*a\*b\*, vibrance
enhancement, custom histogram equalization (whole-image and foreground-only), Sobel filtering
(three implementations), custom image zoom (nearest-neighbor / bilinear), grabCut segmentation
with background blur, and bilateral filtering (OpenCV vs. Gaussian vs. custom).

## Repo layout

```
notebook/230203G_a01.ipynb   - the graded deliverable: run top-to-bottom, then
                                File > Download as > PDF (or `jupyter nbconvert --to pdf`)
code/                         - the same solutions as standalone, per-question .py scripts
                                (handy for reading/review outside Jupyter)
images/                       - input images used by the notebook/scripts
report/                       - an additional hand-written LaTeX report covering the same
                                content (NOT a substitute for the notebook-exported PDF —
                                see note below)
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebook/230203G_a01.ipynb   # run all cells
jupyter nbconvert --to pdf notebook/230203G_a01.ipynb
```

