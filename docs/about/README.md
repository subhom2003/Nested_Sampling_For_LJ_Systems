# About the LJNS implementation

This directory documents the scientific model and the software that
implements it. The code is a bulk Lennard-Jones fluid calculation in a
periodic cubic box, using NVT nested sampling independently at each density.

## Guide

1. [Theory](theory.md)
   - Reduced LJ units and the model Hamiltonian
   - Periodic boundaries, minimum image, and cutoff treatment
   - Nested-sampling volume compression and weights
   - Constrained Metropolis sampling
   - Thermodynamic estimators and phase-boundary proxy
2. [Implementation](implementation.md)
   - Execution graph and module responsibilities
   - Initialization, live points, dead points, and archive formats
   - Parallelism, random-number streams, vectorization, and plotting
3. [Data and workflow](data-and-workflow.md)
   - Presets and CLI usage
   - Output files and observable names
   - Convergence checks, reproducibility, and limitations

The root [README](../../README.md) is the short GitHub-facing entry point.
