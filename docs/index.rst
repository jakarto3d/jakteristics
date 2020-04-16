Jakteristics
============

Jakteristics is a python package to compute point cloud geometric features. 

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

.. _`Contour detection in unstructured 3D point clouds`: https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/timo-jan-cvpr2016.pdf

.. toctree::
   :maxdepth: 2
   
   installation.rst
   usage.rst
   reference.rst
