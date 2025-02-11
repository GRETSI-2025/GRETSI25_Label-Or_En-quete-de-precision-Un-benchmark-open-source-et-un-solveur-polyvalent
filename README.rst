
My Benchopt Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible comparisons of optimization methods.
This benchmark is dedicated to solvers of the Graphical Lasso estimator (Banerjee et al., 2008):


$$\\min_{\\Theta \\succ 0} - \\log \\det (\\Theta) + \\langle \\Theta, S \\rangle + \\alpha \\Vert \\Theta \\Vert_{1,\\mathrm{off}},$$

where $\\Theta$ is the optimization variable and $S$ is the empirical covariance matrix.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/Perceptronium/benchmark_graphical_lasso
   $ benchopt run benchmark_graphical_lasso

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_graphical_lasso -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/Perceptronium/benchmark_graphical_lasso/actoiworkflows/main.yml/badge.svg
   :target: https://github.com/Perceptronium/benchmark_graphical_lasso/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
