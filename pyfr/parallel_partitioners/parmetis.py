# -*- coding: utf-8 -*-

import numpy as np

from pyfr.parallel_partitioners.base import BaseParallelPartitioner

class ParMETISPartitioner(BaseParallelPartitioner):
    name = 'parmetis'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _partition_graph(self, graph, np):
        pass

