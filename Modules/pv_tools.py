import numpy as np
import seissolxdmf as seisx
import pyvista as pv 


##############################################################

# Mesh dealings and enrichment
class SeisSol2pyVista:
    def Mesh(xdmfFilename):
        print("Converting SeisSol mesh to pyvista mesh")

        sx = seisx.seissolxdmf(xdmfFilename)

        ndt = sx.ReadNdt()-2
        xyz = sx.ReadGeometry()
        connect = sx.ReadConnect()

        print("Connectivity shape:",connect.shape)
        print("ndt:",ndt)

        #num_face_nodes = connect.shape[1]

        #leading_values = np.full((connect.shape[0], 1), int(num_face_nodes),dtype=np.uint64)
        #connectivity_w_faces = np.hstack((leading_values, connect))

        mesh = pv.UnstructuredGrid( {pv.CellType.TRIANGLE:connect}, xyz) #pv.PolyData(xyz, connectivity_w_faces)

        return mesh, ndt
    
    def Get_ndt_only(xdmfFilename):
        print("Reading SeisSol mesh 4 ndt")

        sx = seisx.seissolxdmf(xdmfFilename)
        ndt = sx.ReadNdt()-2

        return ndt

    def VoluMesh(xdmfFilename):
        print("Converting SeisSol mesh to pyvista mesh")

        sx = seisx.seissolxdmf(xdmfFilename)

        ndt = sx.ReadNdt()-2
        xyz = sx.ReadGeometry()
        connect = sx.ReadConnect()

        print("Connectivity shape:",connect.shape)
        print("ndt:",ndt)

        # Create the unstructured grid
        mesh = pv.UnstructuredGrid( {pv.CellType.TETRA:connect}, xyz)


        return mesh, ndt

    def FieldEnrichment(xdmfFilename, mesh, field_list, idt):
        print("Converting SeisSol field to pyvista mesh")
        
        sx = seisx.seissolxdmf(xdmfFilename)
        print("Available fields",sx.ReadAvailableDataFields())
        for field in field_list:
            mesh[field] = sx.ReadData(field,idt=idt)

    def Get_Tag(xdmfFilename):
        sx = seisx.seissolxdmf(xdmfFilename)
        return sx.Read1dData("fault-tag",sx.ReadNElements(), isInt=True)






class pv_processing:
    def GetEdges(Mesh, ExtractKwargs={}):
        print("Get edges")

        defaultExtract = dict(non_manifold_edges=False, boundary_edges=True, manifold_edges=False,feature_edges=False)

        defaultExtract.update(ExtractKwargs)

        return Mesh.extract_feature_edges(**defaultExtract)

    def GenerateProjection(Mesh, **kwargs):
        print("projecting onto plane")
        return Mesh.project_points_to_plane(**kwargs)


    def compute_mesh_normals(coord, cells):
        """ Function the normal vectors

        Function 'compute_mesh_normals' computes normals at cells, and at points weighted by the
        area of the surrounding cells on a triangular mesh.

        Args:
            coords (array): np.ndarray(n, k) List of n k-dimensional coordinate points
            cells (array): np.ndarray(m, 3) Triangular cell connectivity

        Returns:
            point_nvecs (array): np.ndarray(n, k) List of k-dimensional normal vector at the n points
        """

        cell_vecs = np.diff(coord[cells], axis=1)
        cell_nvecs = np.cross(-cell_vecs[:, 0, :], cell_vecs[:, 1, :]) / 2.

        return normalize(cell_nvecs)

    def AddLocalFrame(TrialFaultMesh, TrialFault_xyz, TrialFault_connect, depthAxis=[0,0,-1], NormalVect_Exp_List = [0,-1,0]):
        CellNormals = compute_mesh_normals(TrialFault_xyz, TrialFault_connect)
    
        ###############################################################################################
        ### Recomputing the normals to be positive following a given direction
        # This is done for consistency along the mesh. Seissol's output has as a convention of positive
        #NormalVect_Exp_List = [0,-1,0] # can be replaced with e.g. overall orientation, something more robust
        NormalVect_Expected = np.array(NormalVect_Exp_List*(np.shape(CellNormals)[0])).reshape(np.shape(CellNormals)[0],np.shape(CellNormals)[1]) 
        NormalNegMask = np.einsum('ij,ij->i',NormalVect_Expected,CellNormals)<0
        CellNormals[NormalNegMask] = -CellNormals[NormalNegMask]
        ###############################################################################################
        

        ZDir = np.array(depthAxis*np.shape(CellNormals)[0]).reshape(np.shape(CellNormals)[0],np.shape(CellNormals)[1])
        TangentStrike = normalize(np.cross(CellNormals, ZDir))
        TangentDip = normalize(np.cross(TangentStrike, CellNormals))


        TrialFaultMesh["Cell-Normals"] = CellNormals
        TrialFaultMesh["Cell-TangentStrike"] = TangentStrike
        TrialFaultMesh["Cell-TangentDip"] = TangentDip
        
        
        return TrialFaultMesh




# Pyvista plottings
class pvPlots:
    def Init_SingleView():
        print("Prepare view orientation")

    def SetCamera(pl, Cam=['xy',90,0],CamZoom=1.0):
        pl.camera_position = Cam[0]; pl.camera.azimuth = Cam[1]; pl.camera.elevation = Cam[2]
        _ = pl.camera.zoom(CamZoom)

    def ViewSpecsSetup(pl):

        _ = pl.add_axes(line_width=1)
        _ = pl.show_bounds(grid='front', location='outer', all_edges=True)



