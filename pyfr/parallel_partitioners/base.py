# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np

from pyfr.mpiutil import get_comm_rank_root

Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])

def decomp_idx(l, n):
    # Return indices that would split a thing of length l into n parts.
    # Shamelessly stolen from NumPy's array_split function:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html
    l_each_section, extras = divmod(l, n)
    section_sizes = ([0] +
                     extras * [l_each_section + 1] +
                     (l - extras) * [l_each_section]
    )
    div_points = np.array(section_sizes).cumsum()
    return tuple(div_points)


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

            # Read in a subset of the ``con_p0`` dataset.
            nf = mesh['con_p0'].shape[1]
            div_points = decomp_idx(nf, self.nparts)[rank:rank+2]
            con = mesh['con_p0'][:,slice(*div_points)].astype('U4,i4,i1,i1')

            # Start building up the graph.
            con_d = defaultdict(list)
            for el, er in zip(*con):
                lid = (el[0], el[1])
                rid = (er[0], er[1])
                con_d[lid].append(rid)
                con_d[rid].append(lid)
            print('rank = {}, con_d = {}'.format(rank, con_d))

            # Decide which elements will be ours. Need to look at the
            # shape point datasets. Those all will be named
            # ``spt_<element type>_p0``. So
            el_d = {}
            print(mesh.partition_info('spt'))
            for tel, nel in mesh.partition_info('spt').items():
                el_d[tel] = decomp_idx(nel[0], self.nparts)

            print("rank = {}, el_d = {}".format(rank, el_d))

            # Now, the next thing to do: push all the data around. I
            # know where it needs to go now, I think. So, I'd need to
            # start receives for all the elements I want, and sends for
            # all the elements I don't want. So that would mean I need
            # to learn a bit about MPI. Wait: do I know who has the
            # data? The sender knows where it will go, and the receiver
            # has no idea. Actually, there'll be multiple senders with
            # data that the receiver needs. So how will it know when to
            # stop waiting for a receive? That's an interesting puzzle.
            # Hmm... is this what MPI_GATHER is for? I think it might
            # be. But does MPI_GATHER require that all processes send
            # the same amount of data? There's also MPI_GATHERV.

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
