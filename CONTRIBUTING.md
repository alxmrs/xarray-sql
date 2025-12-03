# Contributing Guide

## Where to start?

Please check out the [issues tab](https://github.com/alxmrs/xarray-sql/issues).
Let's have a discussion over there before proceeding with any changes. Great
minds think alike -- someone may have already created an issue related to your
inquiry. If there's a bug, please let us know.

If you're totally new to open source development, I recommend
reading [Xarray's contributing guide](https://docs.xarray.dev/en/stable/contributing.html).

## Developer setup

0. We use `uv` to manage the project: https://docs.astral.sh/uv/getting-started/installation/
1. Clone the repository (bonus: [via SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account))
   and `cd xarray_sql` (the project root).
2. Install dev dependencies via: `uv sync --dev`
3. Install pre-commit hooks: `uv run pre-commit install`

   This will automatically run code formatting (pyink) and type checking (mypy) before each commit.
   You can also run the hooks manually with: `uv run pre-commit run --all-files`


## Before submitting a pull request...

Thanks so much for your contribution! For a volunteer led project, we so
appreciate your help. A few things to keep in mind:
- Please be nice. We assume good intent from you, and we ask you to do the same for us.
- Development in this project will be slow if not sporadic. Reviews will come
  as time allows.
- Every contribution, big or small, matters and deserves credit.

Here are a few requests for your development process:
- We require all code to be formatted with `pyink` and type-checked with `mypy`.
  These checks run automatically via pre-commit hooks (see Developer setup above).
  If you need to run them manually:
  - Formatting: `uv run pre-commit run pyink --all-files` or `uvx pyink .`
  - Type checking: `uv run pre-commit run mypy --all-files` or `uv run mypy xarray_sql/`
- Please include unit tests, if possible, and performance tests when you touch the core functionality (see `perf_tests/`).
- It's polite to do a self review before asking for one from a maintainer. Don't stress if you forget; we all do sometimes.
- Please add (or update) documentation when adding new code. We use [Google Style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- We are thrilled to get documentation-only PRs -- especially spelling and typo fixes (I am a bad speller). If writing tutorials excites you, it would be to everyone's benefit.
