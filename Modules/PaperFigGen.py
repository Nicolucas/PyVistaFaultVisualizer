import numpy as np
import seissolxdmf as seisx
import pyvista as pv 
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pv_tools as pvt


R_param = lambda PnArray,TArray, mu_s, mu_d : (np.abs(TArray) - mu_d*np.abs(PnArray))/((mu_s-mu_d)*np.abs(PnArray))
S_param = lambda PnArray,TArray, mu_s, mu_d : (mu_s*np.abs(PnArray)-np.abs(TArray))/(np.abs(TArray) - mu_d*np.abs(PnArray))


def TvMuSn(Pn0,T0,mu_s,Mu_d):
    return np.abs(T0) - mu_s*np.abs(Pn0)
    
def tagwise_SR(Mesh,mu_s,mu_d,taglist):
    tagMask = np.isin(Mesh['FaultTag'], taglist)

    if 'S' not in Mesh.array_names:
        Mesh['S'] = np.full(Mesh["Pn0"].shape[0], np.nan)
    if 'R' not in Mesh.array_names:
        Mesh['R'] = np.full(Mesh["Pn0"].shape[0], np.nan)
    if 'TvMuSn' not in Mesh.array_names:
        Mesh['TvMuSn'] = np.full(Mesh["Pn0"].shape[0], np.nan)

    Mesh["S"][tagMask]  = S_param(Mesh["Pn0"][tagMask],np.abs(Mesh["T0"][tagMask]),mu_s,mu_d)
    Mesh["R"][tagMask]  = R_param(Mesh["Pn0"][tagMask],np.abs(Mesh["T0"][tagMask]),mu_s,mu_d)
    Mesh["TvMuSn"][tagMask]  = TvMuSn(Mesh["Pn0"][tagMask],np.abs(Mesh["T0"][tagMask]),mu_s,mu_d)

    
    print("Max Mu_s for tag:",str(taglist), np.max(np.abs(Mesh["T0"][tagMask])/np.abs(Mesh["Pn0"][tagMask])))



def StressDrop(sx, SlipThreshold = 0.01, timeIndices = None):
    if timeIndices is None:
        timeIndices = [0, sx.ReadNdt()] # Only one event

    Ts0 = sx.ReadData('Ts0')[timeIndices[0]:timeIndices[1]]
    Td0 = sx.ReadData('Td0')[timeIndices[0]:timeIndices[1]]
    shearStress = np.sqrt(Ts0**2 + Td0**2)


    ASl = sx.ReadData('ASl')
    ASl = ASl[timeIndices[0]:timeIndices[1]] - ASl[timeIndices[0]]

    # Consider only slipped patch for more than a threshold
    idx = np.where(ASl[-1,:] > SlipThreshold)[0]


    # Filtered shear stress and slip
    StressDropAllCells = np.zeros(np.shape(ASl[-1,:]))

    StressDropAllCells[idx] =  np.sum(shearStress[:-1, idx] - shearStress[1:, idx], axis=0)

    return StressDropAllCells


def AddMeshField(pl,Mesh,Edges,field, **kwargs):
    pl.add_mesh(Mesh,scalars=field, lighting=False, show_scalar_bar=False, **kwargs)
    pl.add_mesh(Edges, lighting=False, color="k")

def SetupView(pl, MeshScaled, Cam=['xz',0,0],CamZoom=3.8, depth_bound = -50):
    _ = pl.show_grid(grid='back',
        location='outer',
        ticks='both',
        n_xlabels=5,
        n_ylabels=3,
        n_zlabels=2,
        fmt='{:.0f}',
        bold=False,
        show_zaxis= False,    show_yaxis=False,     show_xaxis=True,
        font_size = 14,
        bounds=(0,1200,MeshScaled.bounds[2],MeshScaled.bounds[3],depth_bound,0),
        xtitle='', ztitle='',use_3d_text=False,
        minor_ticks=False,
        )

    pvt.pvPlots.SetCamera(pl, Cam=Cam,CamZoom=CamZoom)

def SetupIsometricView(pl, MeshScaled, Cam=['xz',0,0],CamZoom=3.8, depth_bound = -50):
    _ = pl.show_grid(grid='back',
        location='outer',
        ticks='both',
        n_xlabels=5,
        n_ylabels=3,
        n_zlabels=3,
        fmt='{:.0f}',
        bold=False,
        show_zaxis= False,    show_yaxis=False,     show_xaxis=True,
        font_size = 14,
        bounds=(0,1200,MeshScaled.bounds[2],MeshScaled.bounds[3],depth_bound,0),
        xtitle='', ztitle='',use_3d_text=False,
        minor_ticks=True,
        )
    pvt.pvPlots.SetCamera(pl, Cam=Cam,CamZoom=CamZoom)
    pl.enable_parallel_projection()


def Filter_Dataset_Around_loc(MeshPoints, Loc):
    tree = KDTree(Loc)
    dist, ids = tree.query(MeshPoints)
    return dist

def GetHypocenterContours(Hypocenter, MeshScaled, isosurfaces):
    MyMesh = MeshScaled.copy(deep=True)

    MyMesh['Hypocenter_Dist'] = Filter_Dataset_Around_loc(MyMesh.points[:,:2], [Hypocenter])

    contours = MyMesh.contour(isosurfaces+MyMesh['Hypocenter_Dist'].min()+5e-1,scalars='Hypocenter_Dist')
    return contours


def WrapperMeshField2Image(Mesh,Edges,field, Cam=['xz',0,20], CamZoom=3.7,depth_bound=-50, **kwargs):

    pl = pv.Plotter(shape=(1, 1),window_size=[2000, 500],off_screen=True)
    AddMeshField(pl,Mesh,Edges,field,**kwargs)
    SetupView(pl, Mesh, Cam=Cam,CamZoom=CamZoom, depth_bound=depth_bound)
    pl.show()

    Image = pl.image.copy()
    pl.close()
    return Image

def Colorbar(fig,ax,mpl_FieldImg, bbox_to_anchor=(0.3,0.45,1,0.5)):
    axins = inset_axes(ax,
                        width="40%",
                        height="10%",
                        loc="upper left",
                        bbox_to_anchor=bbox_to_anchor,
                        bbox_transform=ax.transAxes)
    fig.colorbar(mpl_FieldImg, shrink=0.3, cax=axins, orientation = "horizontal")
    return axins

def PLT_cbar_Annotations(fig,ax,mpl_FieldImg,timestep,ColorLabel, zlabel_pos=(0,160), xlabel_pos=(950,400)):
    axins = Colorbar(fig,ax,mpl_FieldImg, bbox_to_anchor=(0.65,0.5,0.8,0.25))
    axins.set_xlabel(ColorLabel)
    axins.xaxis.set_label_position('top')

    xlabel = "x (km)"
    zlabel = "z (km)" 

    ax.annotate(zlabel,zlabel_pos, xytext=zlabel_pos)
    ax.annotate(xlabel,xlabel_pos, xytext=xlabel_pos)
    ax.annotate(f"Time (s): {timestep}",(0,130), xytext=(0,130))

    ax.axis(False)


def YieldCohesion(x,y):
    c0 = -1e6
    c0Profile  = np.where(x>-5000,c0-20000*(5000-np.abs(x)), c0)

    return -c0Profile +y


def SaveFig(fig,path,model,field):
    FileName = model + "_"  + field

    fig_kwarg = dict(dpi=300, bbox_inches="tight")

    fig.savefig(path+FileName+'.png', **fig_kwarg)
    fig.savefig(path+FileName+'.pdf', **fig_kwarg)
    fig.savefig(path+FileName+'.eps', **fig_kwarg)


######################################################################## 
# Front view Fault
########################################################################
ColorLabelMapping = dict(ASl="Accumulated Slip (m)",
                         Ts0=r"$\tau_s$ (MPa)",
                         Td0=r"$\tau_d$ (MPa)",
                         Pn0=r"$\sigma_n$ (MPa)",
                         RT = "Rupture time (s)",
                         Ss = r"$S_s = (\mu_s|\sigma_n|-|\tau_s|)/(|\tau_s|-\mu_d|\sigma_n|)$",
                         Sd = r"$S_d = (\mu_s|\sigma_n|-|\tau_d|)/(|\tau_d|-\mu_d|\sigma_n|)$",
                         Rs = r"$R_s = (|\tau_s|-\mu_d|\sigma_n|)/((\mu_s-\mu_d)|\sigma_n| )$" ,
                         Rd = r"$R_d = (|\tau_d|-\mu_d|\sigma_n|)/((\mu_s-\mu_d)|\sigma_n| )$" ,
                         R  = r"$R = (|\tau|-\mu_d|\sigma_n|)/((\mu_s-\mu_d)|\sigma_n| )$" ,
                         S  = r"$S = (\mu_s|\sigma_n|-|\tau|)/(|\tau|-\mu_d|\sigma_n|)$",
                         Vr = "Rupture velocity (km/s)",
                         SRs = "Slip rate along strike (m/s)",
                         SRd = "Slip rate along dip (m/s)",
                         SR = "Slip rate (m/s)",
                         StressDrop = "Stress drop (MPa)",
                         FaultTag = "Fault Tag"
                         )

def WrapperMeshField2Image_Iso(Mesh,Edges,field, Cam=['xz',0,0], CamZoom=3.7,depth_bound=-50,**kwargs):

    pl = pv.Plotter(shape=(1, 1),window_size=[2000, 500],off_screen=True)
    AddMeshField(pl,Mesh,Edges,field,**kwargs)
    SetupIsometricView(pl, Mesh, Cam=Cam,CamZoom=CamZoom, depth_bound=depth_bound)
    pl.show()

    Image = pl.image.copy()
    pl.close()
    return Image


def PLT_SingleFaultField(field, MeshScaled, Edges, LabelkwargsUp={}, **kwargs):
    Labelkwargs = dict(timestep=0, nuc_pixel_coordinates = [1265,230],depth_bound=-50,zlabel_pos=(25,190),xlabel_pos=(950,360))
    Labelkwargs.update(LabelkwargsUp)
    ImageField = WrapperMeshField2Image_Iso(MeshScaled,Edges,field,depth_bound=Labelkwargs["depth_bound"], **kwargs)


    fig = plt.figure(figsize=(24, 4)) 
    ax = fig.add_subplot()
    mpl_FieldImg = ax.imshow(ImageField,**kwargs)

    PLT_cbar_Annotations(fig,ax,mpl_FieldImg,Labelkwargs["timestep"],ColorLabelMapping[field],zlabel_pos=Labelkwargs["zlabel_pos"],xlabel_pos=Labelkwargs["xlabel_pos"])

    ############################################################################
    # Nucleation marker
    ############################################################################
    nuckwargs = dict(marker='*', s=100, color="yellow",ec="k",lw=.5)
    ax.scatter(*Labelkwargs["nuc_pixel_coordinates"], **nuckwargs)

    return  fig,ax