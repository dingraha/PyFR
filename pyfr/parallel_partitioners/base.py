# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict, namedtuple, defaultdict
from itertools import cycle
import re
import uuid

import numpy as np

from pyfr.mpiutil import get_comm_rank_root

Graph = namedtuple('Graph', ['vdist', 'vtab', 'etab', 'vwts', 'ewts'])


def split(Ntotal, Nsections):
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

    def _read_partial_con(self, mesh, rank):
        # Number of face connections.
        nf = mesh['con_p0'].shape[1]
        if rank == 0:
            print("nf = {}".format(nf))

        # The min and max connections this MPI process will read in.
        div_points = split(nf, self.nparts)[rank:rank+2]

        # Read the connectivity.
        con = mesh['con_p0'][:,slice(*div_points)].astype('U4,i4,i1,i1')

        return con


    def _distribute_con(self, mesh, comm, rank, root):
        nparts = self.nparts

        # Read in a subset of the ``con_p0`` dataset.
        con = self._read_partial_con(mesh, rank)

        # Duplicate the connections (b -> a for each a -> b).
        con = np.hstack([con, con[::-1]])

        # Decide which elements belong to each MPI rank (Element Type to
        # Element Range MAP).
        etrermap = OrderedDict()
        part_info = mesh.partition_info('spt')
        for et in sorted(part_info.keys()):
            etrermap[et] = split(part_info[et][0], nparts)

        if rank == root:
            print("etrermap = {}".format(etrermap))

        # So, now each MPI process will distribute the interface
        # information. 
        con_ret = []
        for brank in range(nparts):
            # Need to sort the keys, since we need each MPI process to
            # call MPI_GATHER in the same order. Unless there's a
            # non-blocking version. Is there? Yes, but only in MPI3, I
            # think. And mpi4py doesn't appear to have it.

            # Loop over each Element Type, sending all the connections
            # that MPI rank `brank` will own.
            for et in etrermap.keys():
                # Min and max element IDs that `brank` will own.
                emin, emax = etrermap[et][brank], etrermap[et][brank+1]

                # Extract the relevent interfaces from con.
                idx = np.all(
                        [
                            con['f0'][0] == et,
                            con['f1'][0] >= emin,
                            con['f1'][0] < emax
                        ], axis=0
                    )
                consend = con[:, idx]

                # Send the connections.
                conrecv = comm.gather(consend, root=brank)

                if rank == brank:
                    # Collapse each MPI process's part of the con_p0
                    # array into one array, and save it.
                    con_ret.append(np.hstack(conrecv))
                
            if rank == brank:
                con_ret = np.hstack(con_ret)
                print('con_ret(rank={}) =\n{}'.format(rank, con_ret))
                
        # All done: RETurn the CONnectivity array.
        return con_ret, etrermap


    def _partition_graph(self, graph, partwts):
        pass

    
    def _construct_partial_graph(self, con, etrermap, rank):
        nparts = self.nparts

        # Partially reverse the mapping of etrermap: rank->element
        # type->element range, instead of element type->rank->element
        # range.
        retermap = [[(et, (er[r], er[r+1])) for et, er in etrermap.items()] for r in range(nparts)]
        retermap = [OrderedDict(x) for x in retermap]
        if rank == 0:
            print("retermap = {}".format(retermap))

        # Get the offsets.
        retoffmap = []
        next_id = 0
        for eter in retermap:
            x = OrderedDict()
            for et, (emin, emax) in eter.items():
                x[et] = next_id - emin
                next_id += emax - emin
            retoffmap.append(x)
        if rank == 0:
            print("retoffmap = {}".format(retoffmap))

        # Get the vdist array, which tells us how the graph verticies
        # are distributed to the MPI ranks.
        et = next(reversed(etrermap))
        vdist = [etermap[et][-1] + etoffmap[et] for etermap, etoffmap in zip(retermap, retoffmap)]
        vdist = np.array([0,] + vdist)
        if rank == 0:
            print("vdist =\n{}".format(vdist))

        # Edges of the dual graph. Duplicate the interfaces, so that
        # there's b->a for every a->b.
        con_l = np.hstack([con, con[::-1]])

        # Sort by the left hand side, first by the element type, and
        # then by element ID.
        idx = np.lexsort([con_l['f0'][0], con_l['f1'][0]])
        con_l = con_l[:, idx]

        # Left and right hand side element types/indicies.
        lhs, rhs = con_l[['f0', 'f1']]

        # Compute vertex offsets.
        vtab = np.where(lhs[1:] != lhs[:-1])[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))
        print("rank = {}: vtab =\n{}".format(rank, vtab))

        # Compute the element type/index to vertex number map. 
        #vetimap = [tuple(lhs[i]) for i in vtab[:-1]]
        #etivmap = {k: v+vdist[rank] for v, k in enumerate(vetimap)}
        #print("rank = {}: vetimap =\n{}".format(rank, vetimap))
        #print("rank = {}: etivmap =\n{}".format(rank, etivmap))

        return None, None


    def partition(self, mesh):

        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']

        # Get the number of MPI ranks.
        comm, rank, root = get_comm_rank_root()

        # Perform the partitioning
        if self.nparts > 1:

            # Check that we just have one partition in the mesh.
            nparts_cur = max(int(re.search(r'\d+$', n).group(0)) for n in mesh if n.startswith('spt')) + 1
            if nparts_cur != 1:
                raise RuntimeError('Mesh has %d partitions, but '
                                   'parallel_partition supports only 1' % (nparts_cur,))

            con, etrermap = self._distribute_con(mesh, comm, rank, root)

            graph, vetimap = self._construct_partial_graph(con, etrermap, rank)

            # Dummy for now.
            newmesh = mesh

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
