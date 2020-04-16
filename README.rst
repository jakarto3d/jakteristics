
Jakteristics
~~~~~~~~~~~~

* **Documentation**: https://jakteristics.readthedocs.io
* **Github**: https://github.com/jakarto3d/jakteristics

Jakteristics is a python package to compute point cloud geometric features. 

A **geometric feature** is a description of the geometric shape around a point based on its 
neighborhood. For example, a point located on a wall will have a high *planarity*.

The features used in this package are described in the paper
`Contour detection in unstructured 3D point clouds`_.
They are computed based on the eigenvalues and eigenvectors:

* Eigenvalue sum
* Omnivariance
* Eigenentropy
* Anisotropy
* Planarity
* Linearity
* PCA1
* PCA2
* Surface Variation
* Sphericity
* Verticality
* Nx, Ny, Nz (The normal vector)

It's inspired from a similar tool in `CloudCompare <https://www.danielgm.net/cc/>`_.

It's implemented in cython using the BLAS and LAPACK scipy wrappers. It can use multiple cpus, 
and the performance is quite good (at least twice as fast as CloudCompare).

.. _`Contour detection in unstructured 3D point clouds`: https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/timo-jan-cvpr2016.pdf


Installation
============

.. code:: bash

    python -m pip install jakteristics


Usage
=====

Refer to the `documentation <https://jakteristics.readthedocs.io/en/latest/usage.html>`_ for more details.


From python
-----------

.. code:: python

    from jakteristics import compute_features

    features = compute_features(xyz, search_radius=0.15)


CLI
---

Once the package is installed, you can use the `jakteristics` command:

.. code:: bash

    jakteristics input/las/file.las output/file.las --search-radius 0.15 --num-threads 4


Run tests
=========

.. code:: bash

    python -m pip install -r requirements-dev.txt
    python setup.py pytest
