from collections import OrderedDict

import numpy as np

from vtk import vtkUnstructuredGrid, vtkPoints, vtkMultiBlockDataSet
from vtkmodules.vtkPVPythonCatalystPython import vtkCPPythonScriptPipeline
from vtk.util.numpy_support import numpy_to_vtk
import paraview
from paraview.vtk.vtkPVCatalyst import vtkCPProcessor, vtkCPDataDescription

from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where
from pyfr.writers.vtk import BaseShapeSubDiv
from pyfr.mpiutil import get_comm_rank_root

from pyfr.plugins.base import BasePlugin


class CatalystPlugin(BasePlugin):
    name = 'catalyst'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Output frequency. The user will decide what this is through
        # the Catalyst script.
        self.nsteps = 1

        # Catalyst script filename.
        self.script = self.cfg.get(cfgsect, 'script')

        # Divisor
        self.divisor = self.cfg.getint(cfgsect, 'divisor')

        # MPI info
        comm, rank, root = get_comm_rank_root()
        self._mpi_size = comm.size
        self._mpi_rank = rank

        self.mesh = intg.system.mesh
        self.dtype = np.float64
        self._init_mysterious_pv_stuff()
        self._get_vtk_mesh(intg)

        self.coProcessor = vtkCPProcessor()

        self.dataDescription = vtkCPDataDescription()
        self.dataDescription.AddInput("input")

        self.dataDescription.GetInputDescriptionByName("input").SetGrid(
            self._vtk_mbds)

        pipeline = vtkCPPythonScriptPipeline()
        pipeline.Initialize(self.script)
        self.coProcessor.AddPipeline(pipeline)

        self.dataDescription.SetTimeData(intg.tcurr, intg.nacptsteps)
        if self.coProcessor.RequestDataDescription(self.dataDescription):
            self._do_solutions(intg.soln)
            self.coProcessor.CoProcess(self.dataDescription)

    def _init_mysterious_pv_stuff(self):
        import sys
        from vtkmodules import vtkPVClientServerCoreCorePython as CorePython
        try:
            from vtkmodules import vtkPVServerManagerApplicationPython as ApplicationPython
        except ImportError:
            paraview.print_error(
                "Error: Cannot import vtkPVServerManagerApplicationPython")

        # Mysterious thingies that let me use plain old python instead
        # of pvpython.
        paraview.options.batch = True
        paraview.options.symmetric = True
        if not CorePython.vtkProcessModule.GetProcessModule():
            pvoptions = None
            if paraview.options.batch:
                pvoptions = CorePython.vtkPVOptions()
                pvoptions.SetProcessType(CorePython.vtkPVOptions.PVBATCH)
                if paraview.options.symmetric:
                    pvoptions.SetSymmetricMPIMode(True)
            ApplicationPython.vtkInitializationHelper.Initialize(
                sys.executable, CorePython.vtkProcessModule.PROCESS_BATCH,
                pvoptions)

    def _get_vtk_mesh(self, intg):
        # Partition number.
        rank = self._mpi_rank

        # Get element types and array shapes. For two element types and
        # one partition, the keys are
        #
        #   ['spt_quad_p0', 'spt_tri_p0']
        #
        # For a four-process parallel case, it's
        #
        # ['spt_tri_p0', 'spt_quad_p1', 'spt_tri_p1', 'spt_tri_p2', 'spt_quad_p3', 'spt_tri_p3']
        #
        self.mesh_inf = OrderedDict()
        for mk, mv in self.mesh.array_info('spt').items():
            prt = int(mk.split('p')[-1])
            if prt == rank:
                self.mesh_inf[mk] = mv

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]

        # Create a multiblock vtk data set.
        self._vtk_mbds = vtkMultiBlockDataSet()
        n_blocks = len(self.mesh.array_info('spt').keys())
        self._vtk_mbds.SetNumberOfBlocks(n_blocks)

        self._vis_fields = {}
        for mk in self.mesh_inf:

            # Get the VTK block number for this partition-element type
            # combination.
            b = list(self.mesh.array_info('spt').keys()).index(mk)
            new_mesh_inf = list(self.mesh_inf[mk]) + [b, ]
            self.mesh_inf[mk] = tuple(new_mesh_inf)

            vtk_ugrid = vtkUnstructuredGrid()
            vtk_points = vtkPoints()

            # name = name of the element type (e.g. 'tri', 'quad')
            name = self.mesh_inf[mk][0]
            mesh = self.mesh[mk].astype(self.dtype)

            # Dimensions
            nspts, neles = mesh.shape[:2]
            # mesh.shape = (number of "standard" points per element(?,
            # like, three for tri, four for quad), number of elements,
            # spatial dimension(?))

            # Sub divison points inside of a standard element
            svpts = self._get_std_ele(name, nspts)
            nsvpts = len(svpts)
            # I think nsvpts is the number of nodes that each element
            # will be split into. svpts is where they will be in the
            # element local coordinate system, I think.

            # Generate the operator matrices
            mesh_vtu_op = self._get_mesh_op(name, nspts, svpts)

            # Calculate node locations of VTU elements
            vpts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
            vpts = vpts.reshape(nsvpts, -1, self.ndims)

            # Append dummy z dimension for points in 2D
            if self.ndims == 2:
                vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

            vpts = vpts.swapaxes(0, 1)
            vpts = vpts.reshape(-1, vpts.shape[-1])
            vtk_points.SetData(numpy_to_vtk(vpts))
            vtk_ugrid.SetPoints(vtk_points)

            # Perform the sub division
            subdvcls = subclass_where(BaseShapeSubDiv, name=name)
            nodes = subdvcls.subnodes(self.divisor)
            # I think nodes is the local element connectivity array,
            # flattened.

            # Prepare VTU cell arrays. vtk_con starts out as a repeat of
            # the element-local connetivity matrix, then gets offset by
            # the global element ID, or something.
            vtu_con = np.tile(nodes, (neles, 1))
            vtu_con += (np.arange(neles)*nsvpts)[:, None]

            # Tile VTU cell type numbers
            vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)

            # Nodes per cell array.
            vtu_npc = np.array(
                [subdvcls.vtk_nodes[t] for t in subdvcls.subcells(self.divisor)])
            vtu_npc = np.tile(vtu_npc, (neles, 1)).reshape((vtu_typ.size,))

            # Connectivity matrix.
            vtu_con.shape = (vtu_typ.size, -1)

            # Set the connectivity information.
            for typ, npc, con in zip(vtu_typ, vtu_npc, vtu_con):
                vtk_ugrid.InsertNextCell(typ, npc, con)

            self._vtk_mbds.SetBlock(b, vtk_ugrid)

        self._do_solutions(intg.soln, first_time=True)

    @memoize
    def _get_shape(self, name, nspts):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, self.cfg)

    @memoize
    def _get_mesh_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    @memoize
    def _get_soln_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

    @memoize
    def _get_std_ele(self, name, nspts):
        return self._get_shape(name, nspts).std_ele(self.divisor)

    def __call__(self, intg):
        self.dataDescription.SetTimeData(intg.tcurr, intg.nacptsteps)
        if self.coProcessor.RequestDataDescription(self.dataDescription):
            self._do_solutions(intg.soln)
            self.coProcessor.CoProcess(self.dataDescription)

    def _do_solutions(self, solutions, first_time=False):

        for mk, solution in zip(self.mesh_inf, solutions):

            # Get the block index, and then the vtkUnstructuredGrid.
            b = self.mesh_inf[mk][2]
            vtk_ugrid = self._vtk_mbds.GetBlock(b)

            name = self.mesh_inf[mk][0]
            mesh = self.mesh[mk].astype(self.dtype)
            soln = solution.swapaxes(0, 1).astype(self.dtype)

            # Dimensions
            nspts, neles = mesh.shape[:2]

            # Sub divison points inside of a standard element
            svpts = self._get_std_ele(name, nspts)
            nsvpts = len(svpts)

            soln_vtu_op = self._get_soln_op(name, nspts, svpts)

            # Pre-process the solution, which means converting from
            # conservative to primitive variables.
            soln = self._pre_proc_fields_soln(soln).swapaxes(0, 1)

            # Interpolate the solution to the vis points
            vsoln = np.dot(soln_vtu_op, soln.reshape(len(soln), -1))
            vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)

            # I think vsoln will have shape n_sol_vars, number of
            # nodes in a subdivided element, number of original
            # elements.

            # Process the various fields. I think this just extracts
            # each flow variable into a list, and puts them in an
            # order consistent with visvarmap.
            fields = self._post_proc_fields_soln(vsoln)

            # Set the solution data.
            visvarmap = self.elementscls.visvarmap[self.ndims]
            pointdata = vtk_ugrid.GetPointData()
            for arr, (fnames, vnames) in zip(fields, visvarmap):
                field_name = "{}_{}".format(mk, fnames)
                if first_time:
                    self._vis_fields[field_name] = arr
                    varr = numpy_to_vtk(arr)
                    varr.SetName(fnames.capitalize())
                    pointdata.AddArray(varr)
                else:
                    self._vis_fields[field_name][...] = arr

    def _pre_proc_fields_soln(self, soln):
        # Convert from conservative to primitive variables
        return np.array(self.elementscls.con_to_pri(soln, self.cfg))

    def _post_proc_fields_soln(self, vsoln):
        # Primitive and visualisation variable maps
        privarmap = self.elementscls.privarmap[self.ndims]
        visvarmap = self.elementscls.visvarmap[self.ndims]

        # Prepare the fields
        fields = []
        for fnames, vnames in visvarmap:
            ix = [privarmap.index(vn) for vn in vnames]

            n_var_components = vsoln[ix].shape[0]
            fields.append(vsoln[ix].T.reshape((-1, n_var_components)))

        return fields
