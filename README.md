# The Impact of MRI Image Quality on Statistical and Predictive Analysis on Voxel-Based Morphology

## About

The Forschungszentrum Jülich Machine Learning Library

It is currently being developed and maintained at the [Applied Machine Learning](https://www.fz-juelich.de/en/inm/inm-7/research-groups/applied-machine-learning-aml) group at [Forschungszentrum Juelich](https://www.fz-juelich.de/en), Germany.


## Overview

This repository contains the scripts, data processing pipelines, and analysis code required to reproduce the results presented in the paper *"The impact of MRI image quality on statistical and predictive analysis on voxel-based morphology."* The study investigates how MRI image quality affects univariate statistical analyses and machine learning-based predictions in voxel-based morphology (VBM). By leveraging three large, publicly available datasets, the paper highlights the importance of image quality and sample size in neuroimaging research.

**Paper Abstract:**

MRI brain scans are affected by image artifacts caused by head motion, influencing derived measures such as brain volume and cortical thickness. This study examines the role of automated image quality assessment (IQA) in controlling for the effects of poor-quality images on statistical and predictive analyses. Key findings include:

- Image quality significantly impacts the detection of sex/gender differences in univariate group comparisons, especially for smaller samples.
- Increasing sample size and image quality improves statistical power in univariate analyses but has a marginal effect on classification accuracy for machine learning approaches.
- For univariate methods, higher image quality is crucial, while machine learning benefits more from larger sample sizes.

[**Paper Link:**](https://arxiv.org/abs/2411.01268) https://arxiv.org/abs/2411.01268

## Repository Structure

- `data/`: Scripts to preprocess and prepare datasets for analysis.
- `code/statistics/`: Python scripts for univariate statistical tests, machine learning experiments, and IQA evaluations.
- `code/sex_classification/`: Python scripts for machine learning sex classification.
- `output/`: Directory to store analysis outputs, figures, and tables.
- `plot/`: Jupyter notebooks providing step-by-step workflows for reproducing key results and figures from the paper.


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/N-Nieto/QC.git
   cd QC
