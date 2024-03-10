# Contributing Guide

## Where to start?

Please check out the [issues tab](https://github.com/alxmrs/qarray/issues).
Let's have a discussion over there before proceeding with any changes. Great
minds think alike -- someone may have already created an issue related to your
inquiry. If there's a bug, please let us know.

If you're totally new to open source development, I recommend
reading [Xarray's contributing guide](https://docs.xarray.dev/en/stable/contributing.html)
.

## Developer setup

0. (Recommended) Create a project-specific python
   environment. [(mini)Conda](https://docs.anaconda.com/free/miniconda/index.html)
   or [Mamba](https://mamba.readthedocs.io/en/latest/)
   is preferred.
1. Clone the repository (bonus: [via SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account))
   and `cd xarray_sql` (the project root).
1. Install dev dependencies via: `pip install -e ".[dev]` 