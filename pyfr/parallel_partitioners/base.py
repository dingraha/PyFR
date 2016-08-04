# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


class BaseParallelPartitioner(object):
    # Approximate element weighting table
    _ele_wts = {'quad': 3, 'tri': 2, 'tet': 2, 'hex': 6, 'pri': 4, 'pyr': 3}

    def __init__(self, np):
        self.np = np

    def _partition_graph(self, graph, partwts):
        pass

    def partition(self, mesh):
        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']

        # Perform the partitioning
        if self.np > 1:

            # Read in a small chunk of the mesh. What I want is for each
            # MPI process to read in an equal chunk of the ``con_p0``
            # thingy. How do I do that? First, need to know my rank.
            # That's the first thing to figure out, then: fire up MPI.
            newmesh = mesh
            
        # Short circuit
        else:
            newmesh = mesh

        # Generate a new UUID for the mesh
        newmesh['mesh_uuid'] = newuuid = str(uuid.uuid4())

        # Build the solution converter
        partition_soln = None

        return newmesh, partition_soln
