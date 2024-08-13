import numpy as np
import matplotlib.colors as mcolors
from palettable.colorbrewer import sequential as cmapC
import palettable.mycarta as cmapaSR
from palettable.cmocean import diverging as cmos
from palettable.scientific import sequential as cmapa

CmapDavos = cmapa.Davos_20_r.mpl_colormap
Cmap2 = cmapC.Greens_9_r.mpl_colormap
Cmap3 = cmapC.Greys_9.mpl_colormap
Cmap5 = cmapC.Blues_9.mpl_colormap
Cmap5_r = cmapC.Blues_9_r.mpl_colormap

CmapSR = cmapaSR.LinearL_20_r.mpl_colormap
CmapSR_r = cmapaSR.LinearL_20.mpl_colormap
cmoDelta = cmos.Delta_20.mpl_colormap
CmapSl = cmapa.Bilbao_20.mpl_colormap
CmapRT = cmapa.Oslo_20_r.mpl_colormap

Cmap_SlipRate = cmapa.LaJolla_20.mpl_colormap

######################################################################## 
# cmap functions & variables
########################################################################

sCmap = lambda mapColor, sampling, a=0.2, b=0.8: mapColor(np.linspace(a,b,sampling))

Cpellet = int(256/8)

######################################################################## 
# cmap combos
########################################################################
colors = np.vstack((sCmap(Cmap5,Cpellet*12,a=0,  b=0.7),
                    sCmap(Cmap3,Cpellet*88,a=0.2,b=0.8),
                    ))

S_param_Cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)


########################################################################
colors = np.vstack((sCmap(Cmap5_r,Cpellet*15,a=0,  b=0.7),
                    sCmap(cmoDelta,Cpellet*150,a=0.5,b=1),
                    ))

R_param_Cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

########################################################################
colors = np.vstack((sCmap(Cmap2,Cpellet*15,a=0.3,  b=1),
                    sCmap(Cmap5,Cpellet*12,a=0.3,  b=1),
                    sCmap(Cmap3,Cpellet*88,a=0.3,b=.8),
                    ))

S2_param_Cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

