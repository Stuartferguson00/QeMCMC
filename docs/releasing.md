# Releasing to PyPI

To publish a new release just **tag a commit on `main` and push the tag.**

I've defined a github workflow ([`publish.yaml`](https://github.com/Stuartferguson00/QeMCMC/blob/main/.github/workflows/publish.yaml)) that gets triggered when a tag starting with `v` is pushed. The package version is derived from the tag by `setuptools_scm`, then the workflow builds the package and uploads it to [PyPI](https://pypi.org/project/qemcmc/).

## 1. How to choose the version

We're gonna follow [semantic versioning](https://semver.org/) in the form `vMAJOR.MINOR.PATCH`
(e.g. `v0.3.0`).

- **PATCH**  — bug fixes only, no API changes.
- **MINOR**  — new features (but should be backwards compatible).
- **MAJOR**  — breaking API changes (or to declare a stable build 1.0).

If you wanna see the last released version:

```bash
git tag --sort=-v:refname | head -1
```

## 2. Tag and push

Make sure you are in `main` and it's up to date (pull the latest changes)

```bash

# Create tag (with whatever version)
git tag v0.6.7

# Push the tag (triggers the publish workflow)
git push origin v0.6.7
```

After this, the Github Actions workflow will automatically build and publish it to PyPI.

## 3. Verify

- Check new version got published [pypi.org/project/qemcmc](https://pypi.org/project/qemcmc/).
- Or can just test with a fresh install or upgrade:

  ```bash
  pip install --upgrade qemcmc
  ```

## Backend details (in case i forget how it works in the future :D)


- **Versioning** — `pyproject.toml` sets `dynamic = ["version"]` with `[tool.setuptools_scm]`,
  so the version is read straight from the git tag at build time. Don't need to manually change version anywhere else apart from the creation of the tag.
- **Publishing** — `publish.yaml` runs on any pushed tag matching `v*` and builds the
  package with `python -m build`.