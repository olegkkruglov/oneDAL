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

Triangle Counting algorithm receives an :capterm:`undirected graph<Undirected graph>` :math:`G` as an input and counts
the number of triangles in :math:`G`. The algorithm can compute per-vertex (local) triangle counts,
the total (global) triangle count, or both simultaneously.

.. |tc_compute|     replace::   :ref:`Computing                   <triangle_counting_compute>`
.. |tc_ordered|     replace::   :ref:`ordered_count               <triangle_counting_ordered_count>`
.. |tc_api|         replace::   :ref:`vertex_ranking(...)          <triangle_counting_t_api>`
.. |tc_api_input|   replace::   :ref:`vertex_ranking_input         <triangle_counting_t_api_input>`
.. |tc_api_result|  replace::   :ref:`vertex_ranking_result        <triangle_counting_t_api_result>`

================ =========================== ============ ================= =================
 **Operation**     **Computational methods**           **Programming Interface**
---------------- --------------------------- ------------------------------------------------
  |tc_compute|         |tc_ordered|            |tc_api|    |tc_api_input|    |tc_api_result|
================ =========================== ============ ================= =================
