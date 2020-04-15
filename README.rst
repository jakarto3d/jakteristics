
Jakteristics
~~~~~~~~~~~~

Documentation: https://jakteristics.readthedocs.io
Github: https://github.com/jakarto3d/jakteristics

Jarakteristics is a python package to compute point cloud geometric features. 

A **geometric feature** is a description of the geometric shape around a point based on its 
neighborhood. For example, a point located on a wall will have a high *planarity*.

The features used in this package are described in the paper
`Contour detection in unstructured 3D point clouds`_.
They are based on the eigenvalues *λ1*, *λ2* and *λ3* and the eigenvectors *e1*, *e2* and *e3*.

* Eigenvalue sum : :math:`λ1 + λ2 + λ3`
* Omnivariance: :math:`(λ1 \cdot λ2 \cdot λ3) ^ {1 / 3}`
* Eigenentropy: :math:`-∑_{i=1}^3 λi  \cdot \ln(λi)`
* Anisotropy: :math:`(λ1 − λ3)/λ1`
* Planarity: :math:`(λ2−λ3)/λ1`
* Linearity: :math:`(λ1−λ2)/λ1`
* PCA1: :math:`λ1/(λ1 + λ2 + λ3)`
* PCA2: :math:`λ2/(λ1 + λ2 + λ3)`
* Surface Variation: :math:`λ3/(λ1+λ2+λ3)`
* Sphericity: :math:`λ3/λ1`
* Verticality: :math:`1-|e3[2]|`
* Nx, Ny, Nz: The normal vector

It's inspired from a similar tool in CloudCompare.

It can use multiple cpus, and the performance is quite good 
(at least twice as fast as CloudCompare).


Installation
============

.. code:: bash

    python -m pip install jakteristics


Usage
=====

Refer to the `documentation <https://jakteristics.readthedocs.io/usage>`_ for more details.


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

`python -m pip install -r requirements-dev.txt`
`python setup.py pytest`
