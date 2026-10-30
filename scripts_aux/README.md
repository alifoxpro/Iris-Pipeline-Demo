# scripts_aux — figure-generation helpers

These are *not* part of the empirical pipeline.  They produce the
illustrative figures used in Chapter 3 (AHE demo, Telea inpainting
demo, Perona-Malik diffusion, Hough voting, Radon transform, Daugman
on real CASIA, system block diagram).

They were originally written under `real_experiments/` with absolute
paths that pointed to the **old** dataset location.  They still work
from there; if you'd rather run them out of this folder, edit the
`CASIA_PATH` constant at the top of each file to:

```python
CASIA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "CASIA-Iris-Thousand"
```

(which is the junction `methodology_implementation/dataset/...`).

| script                              | produces                |
|-------------------------------------|-------------------------|
| `generate_ahe_figure.py`            | `figure_3_9.png`        |
| `generate_inpainting_figure.py`     | `figure_3_10.png`       |
| `generate_diffusion_figure.py`      | `figure_3_11.png`       |
| `generate_segmentation_figures_3.py`| `figure_3_12/13/14.png` |
| `generate_system_diagram.py`        | `figure_3_1.png`        |
