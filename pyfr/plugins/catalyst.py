import numpy as np

from vtk import vtkUnstructuredGrid, vtkPoints
from vtkmodules.vtkPVPythonCatalystPython import vtkCPPythonScriptPipeline
from vtk.util.numpy_support import numpy_to_vtk
import paraview
from paraview.vtk.vtkPVCatalyst import vtkCPProcessor, vtkCPDataDescription

from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where
from pyfr.writers.vtk import BaseShapeSubDiv

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

        self.mesh = intg.system.mesh

        self.dtype = np.float64

        self._init_mysterious_pv_stuff()

        self._get_vtk_mesh()

        self.coProcessor = vtkCPProcessor()

        self.dataDescription = vtkCPDataDescription()
        self.dataDescription.AddInput("input")

        self.dataDescription.GetInputDescriptionByName("input").SetGrid(
            self._vtk_ugrid)

        pipeline = vtkCPPythonScriptPipeline()
        pipeline.Initialize(self.script)
        self.coProcessor.AddPipeline(pipeline)

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

    def _get_vtk_mesh(self):
        # Partition number.
        p = 0

        self._vtk_ugrid = vtkUnstructuredGrid()
        self._vtk_points = vtkPoints()

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]

        self._vtk_vars = list(self.elementscls.visvarmap[self.ndims])

        # Assuming we just have one partition.
        mk = list(self.mesh_inf.keys())[p]

        name = self.mesh_inf[mk][p]
        mesh = self.mesh[mk].astype(self.dtype)

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = self._get_std_ele(name, nspts)
        nsvpts = len(svpts)

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
        for i, p in enumerate(vpts):
            self._vtk_points.InsertPoint(i, p)

        # Perform the sub division
        subdvcls = subclass_where(BaseShapeSubDiv, name=name)
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare VTU cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

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
            self._vtk_ugrid.InsertNextCell(typ, npc, con)
        self._vtk_ugrid.SetPoints(self._vtk_points)

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

            # Partition number.
            p = 0

            # mesh_inf is an OrderedDict. Looks like there's one entry
            # per partition.
            mk = list(self.mesh_inf.keys())[p]

            name = intg.system.ele_types[p]
            mesh = self.mesh[mk].astype(self.dtype)
            soln = intg.soln[p].swapaxes(0, 1).astype(self.dtype)

            # Dimensions
            nspts, neles = mesh.shape[:2]

            # Sub divison points inside of a standard element
            svpts = self._get_std_ele(name, nspts)
            nsvpts = len(svpts)

            soln_vtu_op = self._get_soln_op(name, nspts, svpts)

            # Pre-process the solution
            soln = self._pre_proc_fields_soln(name, mesh, soln).swapaxes(0, 1)

            # Interpolate the solution to the vis points
            vsoln = np.dot(soln_vtu_op, soln.reshape(len(soln), -1))
            vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)

            # Process the various fields
            fields = self._post_proc_fields_soln(vsoln)

            # Set the solution data.
            visvarmap = self.elementscls.visvarmap[self.ndims]
            pointdata = self._vtk_ugrid.GetPointData()
            fields = [arr.T.reshape((-1, arr.shape[0])) for arr in fields]
            for arr, (fnames, vnames) in zip(fields, visvarmap):
                varr = numpy_to_vtk(arr)
                varr.SetName(fnames)
                pointdata.AddArray(varr)

            self.coProcessor.CoProcess(self.dataDescription)

    def _pre_proc_fields_soln(self, name, mesh, soln):
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

            fields.append(vsoln[ix])

        return fields
