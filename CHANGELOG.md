# Changelog

All notable changes to this project should be documented in this file.

The format follows the spirit of Keep a Changelog, and release versions should
match the versions published to PyPI.

## [Unreleased]

- Documented the provenance and implementation differences of every built-in
  steering method, and exposed the same references through registry metadata.
- Clarified current runtime limits and corrected the pandas installation hint
  to use the published ``pysteer-adaptation`` distribution name.
- Removed obsolete prompt-routed and adaptive method implementations and their
  public helpers.
- Added stricter training-data validation for empty datasets, class balance,
  and grouped contrastive methods.
- Added clearer runtime tensor shape validation for steering strategies.
- Made the public `pysteer` facade importable before optional runtime objects
  construct an executor, with clearer pandas dependency errors.
- Relaxed runtime dependency pins to compatible ranges for library installs.
- Standardized validation messages and removed generated IDE/cache artifacts
  from the tracked source set.

## [0.1.1] - 2026-05-28

- Prepared repository metadata, contribution files, and packaging checks for
  public PyPI and GitHub use.
- Added a `pysteer` package facade for `from pysteer import Executor`.
