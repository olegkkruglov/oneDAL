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

.. _api_triangle_counting:

=================
Triangle Counting
=================

.. include::  ../../../includes/graph/triangle-counting-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Triangle Counting <alg_triangle_counting>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::preview::triangle_counting`` namespace and
available via inclusion of the ``oneapi/dal/algo/triangle_counting.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::preview::triangle_counting::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::preview::triangle_counting::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::preview::triangle_counting::task

Enum classes
~~~~~~~~~~~~
.. onedal_enumclass:: oneapi::dal::preview::triangle_counting::kind

.. onedal_enumclass:: oneapi::dal::preview::triangle_counting::relabel

.. _triangle_counting_t_api:

Computing :cpp:expr:`preview::vertex_ranking(...)`
--------------------------------------------------

.. _triangle_counting_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::preview::triangle_counting::vertex_ranking_input

.. _triangle_counting_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::preview::triangle_counting::vertex_ranking_result

Operation
~~~~~~~~~

.. function:: template <typename Graph, typename Descriptor> \
              triangle_counting::vertex_ranking_result preview::vertex_ranking( \
                                         const Descriptor& desc, \
                                         const Graph& g)

   :param desc: Triangle Counting algorithm descriptor :expr:`triangle_counting::descriptor`
   :param g: Input graph

--------
Examples
--------

.. include:: ../../../includes/graph/triangle-counting-examples.rst
