# Project: MBHBS Search (Massive Binary Black Hole Search)

## Core Mission

This project is dedicated to the search for Massive Binary Black Hole (MBHB) coalescences, or similar transient gravitational wave events, using advanced signal processing and data analysis techniques. The primary goal is to efficiently detect and characterize these events within gravitational wave detector data. This involves:

*   **Data Processing:** Handling raw detector data, including downsampling and filtering.
*   **PSD Estimation:** Estimating Power Spectral Densities (PSDs) to characterize detector noise.
*   **Template Bank Generation:** Creating banks of theoretical waveform templates for matched filtering.
*   **Matched Filtering & Search:** Performing optimal matched filtering against data to identify candidate events (triggers) and analyze their significance.
*   **Coincidence Analysis:** Merging triggers from multiple detectors to enhance confidence and pinpoint events.
*   **Optimal SNR Calculation:** Calculating and optimizing Signal-to-Noise Ratio (SNR) for detection.

## Top-Down Structure Report

The project is organized into several key directories, each serving a distinct purpose in the MBHB search pipeline.

### `/hwilas` (Python Package - Core Libraries and Utilities)

This directory contains the core Python libraries and utilities used across the project. It's structured as a Python package, intended for reusability and clean API definitions.

*   `__init__.py`: Package initialization.
*   `utils.py`: General utility functions.
*   `types/`: Defines custom data types, such as for `argparse`.
*   `data_processing/`: Core Python modules for data manipulation, including 'mojito' and 'sangria' specific logic.
*   `psd_estimation/`: Core Python modules for Power Spectral Density estimation, including smoothing algorithms.
*   `search_utils/`: Utilities specifically for search algorithms.

### `/tools` (Python Scripts - Executable Pipeline Components)

This directory contains executable Python scripts that implement various stages of the MBHB search pipeline. These scripts often leverage the libraries defined in `/hwilas`.

*   `data_processing/`: Scripts for processing raw data.
*   `psd_estimation/`: Scripts for estimating PSDs.
*   `search/`: Scripts for performing the gravitational wave search.
*   `template_banks/`: Scripts for generating and manipulating template banks.

### `/examples` (Shell Scripts - Workflow Examples and Demos)

This directory provides example shell scripts demonstrating how to run various components of the pipeline. These are useful for understanding typical workflows and for testing.

*   `data_processing/`: Example shell scripts for data processing workflows.
*   `psd_estimation/`: Example shell scripts for PSD estimation workflows.
*   `search/`: Example shell scripts for search workflows.
*   `template_banks/`: Example shell scripts for template bank workflows.
*   Other root-level scripts in this directory provide miscellaneous examples.

### Root Level Files

*   `.gitignore`: Specifies intentionally untracked files to ignore.
*   `environment.yaml`: Conda environment definition for dependency management.
*   `pyproject.toml`: Project metadata and build system configuration (e.g., poetry, flit).
*   `README.md`: General project information and setup instructions.

## Development Guidelines

*   **Licensing:** All dependencies must be compatible with the project's license. Specifically, no new code with a GPL or AGPL license should be added.
*   **PyCBC Deprecation:** The `pycbc` library is a dependency that is to be removed as soon as a suitable replacement is found. Do not add new functionality that relies on it.

## Agent Interaction Guidelines

*   **Persona:** Any agent working on the `hwilas` project should refer to themselves in the third person as "Wee Lad, the hwilas agent".
