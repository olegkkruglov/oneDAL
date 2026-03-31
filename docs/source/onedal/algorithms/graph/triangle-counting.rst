.. Copyright 2020 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. default-domain:: cpp

.. _alg_triangle_counting:

=================
Triangle Counting
=================

.. include::  ../../../includes/graph/triangle-counting-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _triangle_counting_compute:

Computing
---------

Given an :capterm:`undirected graph<Undirected graph>` :math:`G = (V, E)`, a triangle is a set of three vertices
:math:`\{u, v, w\} \subseteq V` such that :math:`(u, v) \in E`, :math:`(v, w) \in E`, and :math:`(u, w) \in E`.

The algorithm computes:

- **Local triangle count**: For each vertex :math:`v \in V`, the number of triangles that contain :math:`v`.
- **Global triangle count**: The total number of triangles in :math:`G`.

.. rubric:: Example

Consider a graph with 5 vertices and edges:
:math:`\{(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (3,4)\}`.

The triangles in this graph are:
:math:`\{0, 1, 2\}`, :math:`\{0, 1, 3\}`, :math:`\{0, 2, 3\}`, and :math:`\{1, 2, 3\}`.

* The global triangle count is 4.
* The local triangle counts are: vertex 0 has 3, vertex 1 has 3, vertex 2 has 3, vertex 3 has 3, vertex 4 has 0.

.. _triangle_counting_ordered_count:

Computation method: *ordered_count*
------------------------------------

The method orders vertices by degree and iterates over edges in a consistent direction
to avoid counting the same triangle multiple times. For each edge :math:`(u, v)` where
:math:`u < v` in the ordering, the algorithm counts the number of common neighbors
of :math:`u` and :math:`v` using set intersection of their adjacency lists.

The method supports different triangle kinds:

- ``undirected_clique``: Counts undirected triangles (cliques of size 3).
- ``directed_cycle``: Counts directed cycles.
- ``directed_closed_triplet``: Counts directed closed triplets.

An optional vertex relabeling step (enabled via the ``relabel`` parameter) reorders vertices
by degree to improve cache locality and reduce computation time.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Triangle Counting <api_triangle_counting>`.

--------
Examples
--------

.. include:: ../../../includes/graph/triangle-counting-examples.rst
