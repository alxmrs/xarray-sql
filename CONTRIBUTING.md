# Contributing Guide

## Where to start?

Please check out the [issues tab](https://github.com/alxmrs/xarray-sql/issues).
Let's have a discussion over there before proceeding with any changes. Great
minds think alike -- someone may have already created an issue related to your
inquiry. If there's a bug, please let us know.

If you're totally new to open source development, I recommend
reading [Xarray's contributing guide](https://docs.xarray.dev/en/stable/contributing.html).

## Developer setup

We support two workflows: **pixi** (recommended) and **uv**.

### With pixi (recommended)

[Pixi](https://pixi.sh) manages both Python and non-Python dependencies (including the Rust toolchain) in isolated environments.

1. Install pixi: https://pixi.sh/latest/#installation
2. Clone the repository and `cd xarray-sql`.
3. Install and build the Rust extension:
   ```bash
   pixi run postinstall-maturin
   ```
4. Run tests:
   ```bash
   pixi run test
   ```
5. Install pre-commit hooks:
   ```bash
   pixi run -e lint lint-install
   ```
6. Build and serve docs locally:
   ```bash
   pixi run -e docs docs
   ```

Pixi environments are defined in `pixi.toml`. Common tasks:

| Command | Description |
|---|---|
| `pixi run test` | Run pytest |
| `pixi run test-coverage` | Run pytest with coverage |
| `pixi run -e lint lint` | Run all pre-commit hooks |
| `pixi run -e docs docs` | Serve docs locally |
| `pixi run -e docs docs-build` | Build docs to `site/` |
| `pixi run -e build build-wheel` | Build a wheel with maturin |

### With uv

0. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
1. Clone the repository and `cd xarray-sql`.
2. Install dev dependencies: `uv sync --dev`
3. Install pre-commit hooks: `uv run pre-commit install`

   This will automatically run code formatting and type checking before each commit.
   You can also run the hooks manually with: `uv run pre-commit run --all-files`


## Before submitting a pull request...

Thanks so much for your contribution! For a volunteer led project, we so
appreciate your help. A few things to keep in mind:

- Please be nice. We assume good intent from you, and we ask you to do the same for us.
- Development in this project will be slow if not sporadic. Reviews will come
  as time allows.
- Every contribution, big or small, matters and deserves credit.

Here are a few requests for your development process:

- We require all code to be formatted and linted. These checks run automatically
  via pre-commit hooks (see Developer setup above). You can run them manually:
  ```bash
  pixi run -e lint lint
  # or with uv:
  uv run pre-commit run --all-files
  ```
- Please include unit tests, if possible, and performance tests when you touch the core functionality (see `perf_tests/`).
- It's polite to do a self review before asking for one from a maintainer. Don't stress if you forget; we all do sometimes.
- Please add (or update) documentation when adding new code. We use [Google Style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- We are thrilled to get documentation-only PRs -- especially spelling and typo fixes (I am a bad speller). If writing tutorials excites you, it would be to everyone's benefit.
