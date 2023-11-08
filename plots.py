"""Metrics for generic plotting.

Functions
---------
plot_metrics(history,metric)
plot_metrics_panels(history, settings)
plot_map(x, clim=None, title=None, text=None, cmap='RdGy')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as ct
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import palettable
from matplotlib.colors import ListedColormap
import metrics
from shapely.errors import ShapelyDeprecationWarning
import warnings
import regions
import base_directories
import os

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150
plt.style.use('seaborn-notebook')
dpiFig = 300


def get_mycolormap():
    import matplotlib.colors as clr
    import palettable

    cmap = palettable.scientific.sequential.get_map("Buda_20",).colors
    del cmap[0:1]
    del cmap[-4:]

    cmap.append((256*.85, 256*.8, 256*.65))
    cmap.append((256*.9, 256*.85, 256*.8))
    cmap.append((256*.9, 256*.9, 256*.9))
    cmap.append((256, 256, 256))

    cmap = cmap[::-1]
    cmap = np.divide(cmap, 256.)
    return clr.LinearSegmentedColormap.from_list('custom', cmap, N=256)


def savefig(filename, dpi=300):
    for fig_format in (".png", ".pdf"):
        plt.savefig(filename + fig_format,
                    bbox_inches="tight",
                    dpi=dpi)


def get_qual_cmap():
    cmap = palettable.colorbrewer.qualitative.Accent_7.mpl_colormap
    cmap = ListedColormap(cmap(np.linspace(0, 1, 11)))
    cmap2 = cmap.colors
    cmap2[6, :] = cmap.colors[0, :]
    cmap2[2:6, :] = cmap.colors[5:1:-1, :]
    cmap2[1, :] = (.95, .95, .95, 1)
    cmap2[0, :] = (1, 1, 1, 1)
    cmap2[5, :] = cmap2[6, :]
    cmap2[6, :] = [0.7945098, 0.49647059, 0.77019608, 1.]
    cmap2 = np.append(cmap2, [[.2, .2, .2, 1]], axis=0)
    cmap2 = np.delete(cmap2, 0, 0)

    return ListedColormap(cmap2)


def plot_targets(target_train, target_val):
    plt.figure(figsize=(10, 2.5), dpi=125)
    plt.subplot(1, 2, 1)
    plt.hist(target_train, np.arange(0, 8, .1))
    plt.title('Training targets')
    plt.subplot(1, 2, 2)
    plt.hist(target_val, np.arange(0, 8, .1))
    plt.title('Validation targets')
    plt.show()


def drawOnGlobe(ax, map_proj, data, lats, lons, cmap='coolwarm', vmin=None, vmax=None, inc=None, cbarBool=True,
                contourMap=[], contourVals=[], fastBool=False, extent='both', alpha=1., landfacecolor="None", cbarpad=.02):

    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons)  # fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons

    #     ax.set_global()
    #     ax.coastlines(linewidth = 1.2, color='black')
    #     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')

    # ADD COASTLINES
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor=landfacecolor,
        edgecolor='k',
        linewidth=.5,
    )
    ax.add_feature(land_feature)

    # ADD COUNTRIES
    # country_feature = cfeature.NaturalEarthFeature(
    #     category='cultural',
    #     name='admin_0_countries',
    #     scale='50m',
    #     facecolor='None',
    #     edgecolor = 'k',
    #     linewidth=.25,
    #     alpha=.25,
    # )
    # ax.add_feature(country_feature)

    #     ax.GeoAxes.patch.set_facecolor('black')

    if fastBool:
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, alpha=alpha)
    #         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, shading='auto')

    if (np.size(contourMap) != 0):
        contourMap_cyc, __ = add_cyclic_point(contourMap, coord=lons)  # fixes white line by adding point
        ax.contour(lons_cyc, lats, contourMap_cyc, contourVals, transform=data_crs, colors='fuchsia')

    if cbarBool:
        cb = plt.colorbar(image, shrink=.45, orientation="horizontal", pad=cbarpad, extend=extent)
        cb.ax.tick_params(labelsize=6)
    else:
        cb = None

    image.set_clim(vmin, vmax)

    return cb, image


def add_cyclic_point(data, coord=None, axis=-1):
    # had issues with cartopy finding utils so copied for myself

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both', length=4, width=2, which='major', color='dimgrey')
#     ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)


def summarize_errors(metrics_dict):
    marker_size = 15
    alpha = .8

    x_plot = metrics.eval_function(metrics_dict["error_climo"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '--', markersize=marker_size, label='climatology baseline',
             color="gray", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_network"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='network',
             color="orange", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_corr"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='corr. baseline',
             color="cornflowerblue", alpha=alpha/3.)

    x_plot = metrics.eval_function(metrics_dict["error_globalcorr"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='global corr. baseline',
             color="seagreen", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_random"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='random baseline',
             color="gray", alpha=alpha)

    plt.ylabel('MAE (K)')
    plt.xlabel('number of analogues averaged')
    plt.xlim(0, np.max(metrics_dict["analogue_vector"])*1.01)

    plt.ylim(.1, 1.)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title('MAE')


def summarize_skill_score(metrics_dict):
    marker_size = 15
    alpha = .8

    x_plot = metrics.eval_function(metrics_dict["error_climo"])
    x_climatology_baseline = x_plot.copy()

    plt.axhline(y=0, linewidth=1, linestyle='-', color="k", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_network"])
    x_plot = 1. - x_plot/x_climatology_baseline
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='masked analog',
             color="orange", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_corr"])
    x_plot = 1. - x_plot/x_climatology_baseline
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='region corr.',
             color="lightskyblue", alpha=alpha)
    
    x_plot = metrics.eval_function(metrics_dict["error_customcorr"])
    x_plot = 1. - x_plot/x_climatology_baseline
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='custom corr.',
             color="cornflowerblue", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_globalcorr"])
    x_plot = 1. - x_plot/x_climatology_baseline
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='global corr.',
             color="mediumblue", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_random"])
    x_plot = 1. - x_plot/x_climatology_baseline
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='random',
             color="gray", alpha=alpha)

    plt.ylabel('skill score')
    plt.xlabel('number of analogues averaged')
    plt.xlim(0, np.max(metrics_dict["analogue_vector"])*1.01)

    plt.ylim(-0.5, 1.0)
    plt.grid(False)
    plt.legend(fontsize=8)
    plt.title('MAE Skill Score')


def plot_interp_masks(fig, settings, weights_train, lat, lon, region_bool=True, climits=None, central_longitude=215.,
                      title_text=None, subplot=(1, 1, 1), cmap=None, use_text=True, edgecolor="turquoise", 
                      cbarBool=True, cbarpad=0.02):

    if cmap is None:
        cmap = get_mycolormap()

    if settings["maskout_landocean_input"] == "land":
        landfacecolor = "k"
    else:
        landfacecolor = "None"

    for channel in [0, 1]:

        if climits is None:
            cmin = 0.  # np.min(weights_train[:])
            cmax = np.max(weights_train[:])
            climits = (cmin, cmax)

        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                             projection=ct.crs.PlateCarree(central_longitude=central_longitude))

        drawOnGlobe(ax,
                    ct.crs.PlateCarree(),
                    weights_train,
                    lat,
                    lon,
                    fastBool=True,
                    vmin=climits[0],
                    vmax=climits[1],
                    cmap=cmap,
                    extent=None,
                    cbarBool=cbarBool,
                    landfacecolor=landfacecolor,
                    cbarpad=cbarpad
                    )
        if region_bool:
            reg = regions.get_region_dict(settings["target_region_name"])
            rect = mpl.patches.Rectangle((reg["lon_range"][0], reg["lat_range"][0]),
                                         reg["lon_range"][1] - reg["lon_range"][0],
                                         reg["lat_range"][1] - reg["lat_range"][0],
                                         transform=ct.crs.PlateCarree(),
                                         facecolor='None',
                                         edgecolor=edgecolor,
                                         color=None,
                                         linewidth=2.5,
                                         zorder=200,
                                         )
            ax.add_patch(rect)
            if settings["target_region_name"] == "north pdo":
                rect = mpl.patches.Rectangle((150, -30),
                                             50,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=100,
                                             )
                # ax.add_patch(rect)
                rect = mpl.patches.Rectangle((125, 5),
                                             180-125,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=101,
                                             )
                # ax.add_patch(rect)

        plt.title(title_text)
        if use_text:
            plt.text(0.01, .02, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
                    + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
                    fontsize=6, color="gray", va="bottom", ha="left", fontfamily="monospace", backgroundcolor="white",
                    transform=ax.transAxes,
                    )

        return ax, climits


def plot_history(settings, history):
    fontsize = 12
    colors = ("#7570b3", "#e7298a")

    best_epoch = np.argmin(history.history["val_loss"])

    plt.figure(figsize=(14, 10))
    # Plot the training and validations loss history.
    plt.subplot(2, 2, 1)
    plt.plot(
        history.history["loss"],
        "-o",
        color=colors[0],
        markersize=3,
        linewidth=1,
        label="training",
    )

    plt.plot(
        history.history["val_loss"],
        "-o",
        color=colors[1],
        markersize=3,
        linewidth=1,
        label="validation",
    )
    try:
        ymin = 0.97*np.min([history.history["val_loss"], history.history["loss"]])
        ymax = 1.025*np.max([history.history["val_loss"][5], history.history["loss"][25]])
    except:
        ymin = 0.
        ymax = 0.05

    plt.ylim(ymin, ymax)
    plt.yscale("log")
    plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
    plt.title("loss during training")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend(frameon=True, fontsize=fontsize)
    plt.tight_layout()

    plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                '_training_history.png', dpi=dpiFig, bbox_inches='tight')
    plt.close()
