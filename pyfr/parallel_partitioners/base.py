# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np

from pyfr.mpiutil import get_comm_rank_root

Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


class BaseParallelPartitioner(object):
    # Approximate element weighting table
    _ele_wts = {'quad': 3, 'tri': 2, 'tet': 2, 'hex': 6, 'pri': 4, 'pyr': 3}

    def __init__(self, nparts):
        self.nparts = nparts

    def _partition_graph(self, graph, partwts):
        pass

    def partition(self, mesh):

        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']
        print('id(mesh) = %d' % id(mesh))

        # Perform the partitioning
        if self.nparts > 1:

            newmesh = mesh

            # I know there's a routine in the original code that
            # combines existing partitions, but I don't want to deal
            # with that in parallel. I'll just check to make sure we're
            # only dealing with a single partition.

            # Get the number of MPI ranks.
            comm, rank, root = get_comm_rank_root()
            print('nparts = %d, comm.size = %d, rank = %d, root = %d' % (self.nparts, comm.size, rank, root))
            if self.nparts != comm.size:
                raise RuntimeError('Asking for %d partitions but running with '
                                   '%d MPI ranks' % (self.nparts, comm.size))

            # Check that we just have one partition in the mesh
            nparts_cur = max(int(re.search(r'\d+$', n).group(0)) for n in mesh if n.startswith('spt')) + 1
            print('nparts_cur = {}'.format(nparts_cur))
            if nparts_cur != 1:
                raise RuntimeError('Mesh has %d partitions, but '
                                   'parallel_partition supports only 1' % (nparts_cur,))

            # Get the total number of elements in the mesh.
            nel = sum([n[0] for et, n in mesh.partition_info('spt').items()])
            #for et, nel in  mesh.partition_info('spt'):
            #    print

        # Short circuit
        else:
            newmesh = mesh

        # This doesn't work right now, I think because I'm using a
        # read-only .pyfrm file, and I'm not copying the old mesh object
        # to a new mesh object.
        # 
        # Generate a new UUID for the mesh
        #newmesh['mesh_uuid'] = newuuid = str(uuid.uuid4())

        # Build the solution converter
        partition_soln = None

        return newmesh, partition_soln
