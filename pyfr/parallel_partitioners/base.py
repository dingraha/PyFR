# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np

from pyfr.mpiutil import get_comm_rank_root

Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


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

    def _read_partial_con(self, mesh):
        # Get the number of MPI ranks.
        comm, rank, root = get_comm_rank_root()

        # Number of face connections.
        nf = mesh['con_p0'].shape[1]
        if rank == 0:
            print("nf = {}".format(nf))

        # The min and max connections this MPI process will read in.
        div_points = split(nf, self.nparts)[rank:rank+2]

        # Read the connectivity.
        con = mesh['con_p0'][:,slice(*div_points)].astype('U4,i4,i1,i1')

        return con


    def _distribute_con(self, mesh):
        nparts = self.nparts

        # Handy MPI stuff.
        comm, rank, root = get_comm_rank_root()

        # Read in a subset of the ``con_p0`` dataset.
        con = self._read_partial_con(mesh)

        # Duplicate the connections (b -> a for each a -> b).
        con = np.hstack([con, con[::-1]])

        # Decide which elements belong to each MPI rank (Element Type to
        # Element Range MAP).
        etermap = {et: split(en[0], nparts) for et, en in mesh.partition_info('spt').items()}

        if rank == 0:
            print("etermap = {}".format(etermap))

        # So, now each MPI process will distribute the interface
        # information. 
        for brank in range(nparts):
            # Need to sort the keys, since we need each MPI process to
            # call MPI_GATHER in the same order. Unless there's a
            # non-blocking version. Is there? Yes, but only in MPI3, I
            # think. And mpi4py doesn't appear to have it.

            # Send all the connections that MPI rank `brank` will own.
            for et in sorted(etermap.keys()):
                # Min and max element IDs that `brank` will own.
                emin, emax = etermap[et][brank], etermap[et][brank+1]

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
                    conrecv = np.hstack(conrecv)
                    print('conrecv(rank={}) =\n{}'.format(rank, conrecv))
                

#       # Build up a map of interfaces.
#       con_d = defaultdict(list)
#       for el, er in zip(*con):
#           lid = (el[0], el[1])
#           rid = (er[0], er[1])
#           con_d[lid].append(rid)
#           con_d[rid].append(lid)

#       con_d_part = defaultdict(list)
#       for base_rank in range(self.nparts):
#           for tel in sorted(el_d.keys()):
#               emin, emax = el_d[tel][base_rank], el_d[tel][base_rank+1]
#               el_to_send = {(et, ei): elems for (et, ei), elems in con_d.items() if ei >= emin and ei < emax and et == tel}

#               ret = comm.gather(el_to_send, root=base_rank)
#               if rank == base_rank:
#                   for cons in ret:
#                       for node, edges in cons.items():
#                           con_d_part[node] += edges


#       print("rank = {}, con_d_part = {}".format(rank, con_d_part))



    def _partition_graph(self, graph, partwts):
        pass

    
    def _construct_partial_graph(self, mesh, rank):

        # Figure out which part of the mesh is ours.
        nf = mesh['con_p0'].shape[1]
        div_points = split(nf, self.nparts)[rank:rank+2]

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

            # Check that we just have one partition in the mesh.
            nparts_cur = max(int(re.search(r'\d+$', n).group(0)) for n in mesh if n.startswith('spt')) + 1
            if nparts_cur != 1:
                raise RuntimeError('Mesh has %d partitions, but '
                                   'parallel_partition supports only 1' % (nparts_cur,))

            # Read in a subset of the ``con_p0`` dataset.
            #nf = mesh['con_p0'].shape[1]
            #div_points = split(nf, self.nparts)[rank:rank+2]
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
            #    el_d[tel] = split(nel[0], self.nparts)

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

            #pgraph, vetimap = self._construct_partial_graph(mesh, rank)
            #print("rank = {}, pgraph = {}, vetimap = {}".format(rank,
            #    pgraph, vetimap))

            self._distribute_con(mesh)

            # Dummy for now.
            newmesh = mesh

        # Short circuit
        #else:
        #    newmesh = mesh

        # This doesn't work right now, I think because I'm using a
        # read-only .pyfrm file, and I'm not copying the old mesh object
        # to a new mesh object.
        # 
        # Generate a new UUID for the mesh
        #newmesh['mesh_uuid'] = newuuid = str(uuid.uuid4())

        # Build the solution converter
        partition_soln = None

        return newmesh, partition_soln
