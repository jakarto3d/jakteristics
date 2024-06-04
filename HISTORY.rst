.. :changelog:

History
-------

Unreleased
----------


0.6.1 (2024-06-04)
------------------


0.6.0 (2023-04-20)
------------------

* add: number_of_neighbors feature
* add: eigenvalues and eigenvectors features


0.5.1 (2023-04-11)
------------------

* fix: computing features when kdtree is not built from the same points for which we want to compute the features
* drop python 3.6, add wheels for python 3.7-3.11 on linux and windows

0.5.0 (2022-01-26)
------------------

* fix: compatibility with latest laspy version (>= 2.1.1, (2.1.0 has a bug))


0.4.3 (2020-09-24)
------------------

* the default value when features can't be computed should be NaN


0.4.2 (2020-04-20)
------------------

* fix extension import statement


0.4.1 (2020-04-17)
------------------

* fix: create parent directories for output file
* fix: rename --num_threads to --num-threads
* fix: require laspy 1.7 for upper case names in extra dimensions


0.4.0 (2020-04-16)
------------------

* first pypi release
* add github actions


0.3.0 (2020-04-14)
------------------

* add feature-names parameter to compute specific features


0.2.0 (2020-04-10)
------------------

* fix windows compilation with openmp
* add example cloudcompare script
* add num_threads cli parameter and help documentation
* write extra dimensions in the correct order


0.1.2 (2020-04-10)
------------------

* Fix tests


0.1.1 (2020-04-10)
------------------

* Fix bug where single precision was used for intermediate variables


0.1.0 (2020-04-10)
------------------

* First release
