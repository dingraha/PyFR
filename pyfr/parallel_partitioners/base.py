# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict, namedtuple
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
        #etermap = {et: split(en[0], nparts) for et, en in mesh.partition_info('spt').items()}
        etermap = OrderedDict()
        part_info = mesh.partition_info('spt')
        for et in sorted(part_info.keys()):
            etermap[et] = split(part_info[et][0], nparts)

        if rank == root:
            print("etermap = {}".format(etermap))

        # So, now each MPI process will distribute the interface
        # information. 
        for brank in range(nparts):
            # Need to sort the keys, since we need each MPI process to
            # call MPI_GATHER in the same order. Unless there's a
            # non-blocking version. Is there? Yes, but only in MPI3, I
            # think. And mpi4py doesn't appear to have it.

            # Loop over each Element Type, sending all the connections
            # that MPI rank `brank` will own.
            for et in etermap.keys():
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
                    # Collapse each MPI process's part of the con_p0
                    # array into one array, and save it.
                    con_ret = np.hstack(conrecv)
                    print('con_ret(rank={}) =\n{}'.format(rank, con_ret))
                
        # All done: RETurn the CONnectivity array.
        return con_ret, etermap


    def _partition_graph(self, graph, partwts):
        pass

    
    def _construct_partial_graph(self, con, etermap, rank):

        # First thing: construct the array that describes how the graph
        # vertices are distributed among the processes.
        etoffmap = [en[-1] for en in etermap.values()]
        etoffmap = np.array([0] + etoffmap).cumsum()
        etoffmap = {et: off for et, off in zip(etermap.keys(), etoffmap)}

        #vdist = [emin + etoffmap[et] for et, (emin, emax) in
        #        etermap.items()]
        vdist = [0,]
        for et, erange in etermap.items():
            for emax in erange[1:]:
                vdist.append(emax + etoffmap[et])
        vdist = np.array(vdist)
        if rank == 0:
            print("vdist =\n{}".format(vdist))

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

            con, etermap = self._distribute_con(mesh, comm, rank, root)

            graph, vetimap = self._construct_partial_graph(con, etermap, rank)

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
