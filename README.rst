===========
fiber-views
===========


a python package for extracting and manipulating "views" of Fiber-seq data

* Free software: MIT license
* Documentation: `ReadTheDocs <https://fiber-views.readthedocs.io>`_

DISCLAIMER
----------

I developed this software for my own use. I make no commitment at this time to maintain it for others. 
I also cannot guarantee its function or accuracy of results.
  

Install
-------

An up to date version of conda is required for installation: 
::

    git clone https://github.com/MorganHamm/fiber-views.git
    cd fiber-views
    conda env create --file=envs/env.yml
    conda activate fiber-views
    pip install -e .


Features
--------

* Access m6A and mCpG methylation, methylation sensitive patches (MSPs), nucleosome patches, all in sparse matrix form
* raw read sequence in matrix form
* Fiber metadata like read quality and arbitrary BAM tags
* Cluster, re-arange, aggregate fibers
* flexible plotting functions


Publications
------------

Fiber-views has been used for analysis and visualization in the following publications:

Bubb, K.L., Hamm, M.O., Tullius, T.W., Min, J.K., Ramirez-Corona, B., Mueth, N.A., Ranchalis, J., Mao, Y., Bergstrom, E.J., Vollger, M.R., Trapnell, C., Cuperus, J.T., Stergachis, A.B., Queitsch, C., 2025. The regulatory potential of transposable elements in maize. Nat. Plants 11, 1181–1192. https://doi.org/10.1038/s41477-025-02002-z

Bohaczuk, S.C., Amador, Z.J., Li, C., Mallory, B.J., Swanson, E.G., Ranchalis, J., Vollger, M.R., Munson, K.M., Walsh, T., Hamm, M.O., Mao, Y., Lieber, A., Stergachis, A.B., 2024. Resolving the chromatin impact of mosaic variants with targeted Fiber-seq. Genome Res. 34, 2269–2278. https://doi.org/10.1101/gr.279747.124




Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
