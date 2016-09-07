# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np

from pyfr.mpiutil import get_comm_rank_root

Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


def decomp_idx(Ntotal, Nsections):
    # Return indices that would split a thing of length l into n parts.
    # Shamelessly stolen from NumPy's array_split function:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] +
                     extras * [Neach_section+1] +
                     (Nsections-extras) * [Neach_section])
    div_points = np.array(section_sizes).cumsum()
    return tuple(div_points)


class BaseParallelPartitioner(object):
    # Approximate element weighting table
    _ele_wts = {'quad': 3, 'tri': 2, 'tet': 2, 'hex': 6, 'pri': 4, 'pyr': 3}

    def __init__(self):
        # Get the number of MPI ranks.
        comm, rank, root = get_comm_rank_root()
        self.nparts = comm.size


    def _partition_graph(self, graph, partwts):
        pass

    
    def _construct_partial_graph(self, mesh, rank):

        # Figure out which part of the mesh is ours.
        nf = mesh['con_p0'].shape[1]
        div_points = decomp_idx(nf, self.nparts)[rank:rank+2]

        # Edges of the dual graph
        con = mesh['con_p0'][:,slice(*div_points)].astype('U4,i4,i1,i1')
        con = np.hstack([con, con[::-1]])
        # Now con is in the same format as before, but the interfaces
        # are duplicated (a -> b and b -> a).

        # Sort by the left hand side
        # I think this sorts the interfaces first by the element ID, and
        # second by element type. Or maybe first by element type, second
        # by element ID.
        idx = np.lexsort([con['f0'][0], con['f1'][0]])
        con = con[:, idx]

        # Left and right hand side element types/indicies
        lhs, rhs = con[['f0', 'f1']]

        # Compute vertex offsets
        vtab = np.where(lhs[1:] != lhs[:-1])[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))

        # Compute the element type/index to vertex number map
        vetimap = [tuple(lhs[i]) for i in vtab[:-1]]
        etivmap = {k: v for v, k in enumerate(vetimap)}

        # Prepare the list of edges for each vertex
        etab = np.array([etivmap[tuple(r)] for r in rhs])

        # Prepare the list of vertex and edge weights
        vwts = np.array([self._ele_wts[t] for t, i in vetimap])
        ewts = np.ones_like(etab)

        return Graph(vtab, etab, vwts, ewts), vetimap


    def partition(self, mesh):

        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']

        # Get the number of MPI ranks.
        comm, rank, root = get_comm_rank_root()

        # Perform the partitioning
        if self.nparts > 1:

            newmesh = mesh

            # I know there's a routine in the original code that
            # combines existing partitions, but I don't want to deal
            # with that in parallel. I'll just check to make sure we're
            # only dealing with a single partition.

            # Check that we just have one partition in the mesh
            nparts_cur = max(int(re.search(r'\d+$', n).group(0)) for n in mesh if n.startswith('spt')) + 1
            if nparts_cur != 1:
                raise RuntimeError('Mesh has %d partitions, but '
                                   'parallel_partition supports only 1' % (nparts_cur,))

            # Read in a subset of the ``con_p0`` dataset.
            #nf = mesh['con_p0'].shape[1]
            #div_points = decomp_idx(nf, self.nparts)[rank:rank+2]
            #con = mesh['con_p0'][:,slice(*div_points)].astype('U4,i4,i1,i1')

            # Start building up the graph.
            #con_d = defaultdict(list)
            #for el, er in zip(*con):
            #    lid = (el[0], el[1])
            #    rid = (er[0], er[1])
            #    con_d[lid].append(rid)
            #    con_d[rid].append(lid)

            ## Decide which elements will be ours. Need to look at the
            ## shape point datasets. Those all will be named
            ## ``spt_<element type>_p0``. So
            #el_d = {}
            #for tel, nel in mesh.partition_info('spt').items():
            #    el_d[tel] = decomp_idx(nel[0], self.nparts)

            #con_d_part = defaultdict(list)
            #for base_rank in range(self.nparts):
            #    for tel in sorted(el_d.keys()):
            #        emin, emax = el_d[tel][base_rank], el_d[tel][base_rank+1]
            #        el_to_send = {(et, ei): elems for (et, ei), elems in con_d.items() if ei >= emin and ei < emax and et == tel}

            #        ret = comm.gather(el_to_send, root=base_rank)
            #        if rank == base_rank:
            #            for cons in ret:
            #                for node, edges in cons.items():
            #                    con_d_part[node] += edges

            #print("rank = {}, con_d_part = {}".format(rank, con_d_part))

            pgraph, vetimap = self._construct_partial_graph(mesh, rank)
            print("rank = {}, pgraph = {}, vetimap = {}".format(rank,
                pgraph, vetimap))


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
