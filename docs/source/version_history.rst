===============
Version History
===============

v2.0.3 - 2026 / 11 / 21
=======================

* Update package deployment.
* Manual has full version history.

v2.0.2 - 2024 / 01 / 12
=======================

* Option to calculate sigma on histograms.
* Update copyright year.

v2.0.1 - 2023 / 05 / 08
=======================

* Fix missing pandas dependency.
* New class to store and manipulate tracking data.
* `Sim.py` adapted to the new class with new plotting functions and labels; obsolete functions removed.
* Fix nearest S finder.

v2.0.0 - 2023 / 03 / 19
=======================

* Major rewrite: loading based on pandas DataFrames with all columns, and much faster twiss loading.
* Main output class renamed from `Output` to `Load`, with 'struct', 'track' and 'optics' loader options.
* Plot class renamed from `Plot` to `Optics` with new optics plotting functions.
* New tracking classes in `Sim.py` including a BDSIM loader for tracking comparison.
* Utility functions for data manipulation, apertures (`getAperture`) and initial energy.
* Mad8 test files and examples for single elements (drift, quadrupole).
* Python 3 now required; package moved to declarative layout.
* New documentation including tracking.
* Old functionality removed.

v1.6.1 - 2021 / 06 / 15
=======================

* Updated beam size plot.

v1.6.0 - 2020 / 06 / 07
=======================

* Python 3 compatibility (print, dict keys, map) while retaining Python 2 support.
* Thick R matrix added to machine plot.
* Venv support.
* Documentation updates and copyright year.

v1.5.0 - 2019 / 02 / 27
=======================

* Combined plot function taking a list of survey files with options per element type.
* Extra echo value function supporting alternative echo formats.
* Plotting improvements.
* Fix Python highlighting colours in the manual for Sphinx 1.8.

v1.4.1 - 2018 / 08 / 22
=======================

* Version and documentation consistency updates.

v1.4.0 - 2018 / 06 / 23
=======================

* Makefile and setup.cfg recipes for bumpversion and PyPI upload.

v1.3 - 2018 / 05 / 27
=====================

* Internal version number and copyright added.

v1.1 - 2018 / 05 / 23
=====================

* Fix licence.

v1.0.2 - 2018 / 04 / 13
=======================

* Removed dependency on pybdsim.
* Licences added.

v1.0.1 - 2018 / 04 / 12
=======================

* Added pybdsim to install requirements.

v1.0 - 2018 / 04 / 11
=====================

* First manual.
* Plotting similar to pymadx with BDSIM colours, right-click element location and no vertical zoom of the machine diagram.
* MATR added to general output; initial energy added.
* fortranformat made a required dependency.

v0.8 - 2017 / 12 / 11
=====================

* Working setup.py and licence update.

v0.7 - 2017 / 01 / 28
=====================

* Initial setup.py.

v0.6 - 2016 / 11 / 02
=====================

* Docstrings for Sphinx documentation.
* Track no longer imported by default to avoid circular dependencies with pybdsim.

v0.5 - 2016 / 08 / 25
=====================

* Moved to `Output` module and fixed aperture sizes.

v0.4.1 - 2016 / 08 / 23
=======================

* Small modifications for XFEL.

v0.4 - 2016 / 08 / 14
=====================

* Mad8 tracking: generation of inrays from BDSIM output, OBSERVE / ARCHIVE output and track output loader.
* Track data matched to samplers from twiss output.
* New IO with echo value and Mad8 input reader; file conversion.
* Nearest object finder.
* Warnings use the Python warnings module so they can be suppressed.

v0.3 - 2015 / 11 / 07
=====================

* Added current monitor, profile monitor, beam loss monitor and cavity elements.
* Code tidy.

v0.2 - 2015 / 09 / 09
=====================

* Print warnings to stderr to avoid interfering with CMake Python module finding.

v0.1 - 2015 / 04 / 21
=====================

* Initial version: general output loading with row and column access.
* Optics, phase advance, aperture and beam line plots.
* Collimator template generator.
