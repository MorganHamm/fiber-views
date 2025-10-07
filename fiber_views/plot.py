#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:33:42 2024

@author: morgan
"""


import numpy as np
# import pandas as pd


import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection, PolyCollection
import matplotlib.transforms as mtrans

import fiber_views as fv


DEFAULT_WIDTH = 0.8


# -----------------------------------------------------------------------------
# Helper functions

def annotate_boundaries(fview):
    """
    Add start and end position columns to the observation metadata of a fiber view.
    
    This function calculates the leftmost and rightmost positions of actual sequence
    data (excluding gaps) for each fiber and adds them as 's_pos' and 'e_pos' columns
    to the obs dataframe.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object to annotate.
    
    Returns
    -------
    None
        The function modifies the fiber view object in place.
    """
    s_pos = []
    e_pos = []
    mask = fview.layers['seq'] != b'-'
    for i in range(fview.shape[0]):
        s_pos.append(np.min(fview.var.pos[mask[i,:]]))
        e_pos.append(np.max(fview.var.pos[mask[i,:]]) + 1)
    fview.obs['s_pos'] = s_pos
    fview.obs['e_pos'] = e_pos
    return(None)


def make_plot_ax(fview, ax=None):
    """
    Create or prepare a matplotlib axis for plotting a fiber view.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object to plot.
    ax : matplotlib.axes.Axes, optional
        An existing axis to use. If None, a new figure and axis will be created.
        The default is None.
    
    Returns
    -------
    matplotlib.axes.Axes
        The prepared axis with xlim and ylim set appropriately for the fiber view.
    """
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_xlim(fview.var.pos[0], fview.var.pos[-1])
    ax.set_ylim(fview.shape[0], -1) # inverted axis, first fiber at top
    return(ax)


# -----------------------------------------------------------------------------
# plotting primitives

def draw_fiber_lines(fview, ax=None, color="#606060"):
    """
    Draw fibers as horizontal lines on a matplotlib axis.
    
    This function draws each fiber in the fiber view as a horizontal line spanning
    from the start to the end of the sequence data. This visualization is suitable
    for fewer than ~150 fibers.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing sequence data.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw on. If None, a new axis will be created.
        The default is None.
    color : str, optional
        The color of the fiber lines. The default is "#606060".
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with fibers drawn as horizontal lines.
    """
    if ax == None:
        ax = make_plot_ax(fview)
    if ('s_pos' not in fview.obs.columns) | ('e_pos' not in fview.obs.columns):
        annotate_boundaries(fview)
    for i in range(fview.shape[0]):
        ax.hlines(y=i, xmin=fview.obs.s_pos.iloc[i], xmax=fview.obs.e_pos.iloc[i],
                 color=color, lw=0.5, zorder=1)
    return(ax)


def draw_fiber_bars(fview, ax=None, color="#d0d0d0", width=DEFAULT_WIDTH):
    """
    Draw fibers as horizontal bars on a matplotlib axis.
    
    This function draws each fiber in the fiber view as a rectangular bar spanning
    from the start to the end of the sequence data. This visualization is suitable
    for fewer than ~150 fibers.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing sequence data.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw on. If None, a new axis will be created.
        The default is None.
    color : str, optional
        The color of the fiber bars. The default is "#d0d0d0".
    width : float, optional
        The width (height) of each bar in axis units. The default is DEFAULT_WIDTH (0.8).
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with fibers drawn as horizontal bars.
    """
    if ax == None:
        ax = make_plot_ax(fview)
    if ('s_pos' not in fview.obs.columns) | ('e_pos' not in fview.obs.columns):
        annotate_boundaries(fview)
    patch_list = []
    for i in range(fview.shape[0]):
        patch = patches.Rectangle((fview.obs.s_pos.iloc[i], i - 0.5 * width), lw=0,
                                  width=fview.obs.e_pos.iloc[i] - fview.obs.s_pos.iloc[i], 
                                  height=width, color=color, zorder=1)
        patch_list.append(patch)
        # ax.add_patch(patch)
    patch_coll = PatchCollection(patch_list, match_original=True)
    ax.add_collection(patch_coll)
    return(ax)


def draw_regions(fview, ax=None, base_name='msp', color="red", width=DEFAULT_WIDTH):
    """
    Draw genomic regions as colored rectangles on a matplotlib axis.
    
    This function visualizes regions (such as nucleosomes or MSPs) as colored
    rectangles overlaid on the fiber view. Each region is drawn at its corresponding
    position along the fiber.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing region data.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw on. If None, a new axis will be created.
        The default is None.
    base_name : str, optional
        The name of the region type to draw (e.g., 'msp', 'nuc', 'fire').
        The default is 'msp'.
    color : str, optional
        The color of the region rectangles. The default is "red".
    width : float, optional
        The width (height) of each rectangle in axis units. The default is DEFAULT_WIDTH (0.8).
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with regions drawn as colored rectangles.
    """
    if ax == None:
        ax = make_plot_ax(fview)
    region_df = fv.tools.make_region_df(fview, base_name=base_name, zero_pos='center')
    patch_list = []
    for i, region in region_df.iterrows():
        patch = patches.Rectangle((region.start, region.row - 0.5 * width), lw=0,
                                  width=region.length, height=width, color=color, zorder=3)
        patch_list.append(patch)
    patch_coll = PatchCollection(patch_list, match_original=True)
    ax.add_collection(patch_coll)
    return(ax)


def draw_mods(fview, ax=None, mod='m6a', width=DEFAULT_WIDTH, color='#000000'):
    """
    Draw base modifications as vertical marks on a matplotlib axis.
    
    This function visualizes base modifications (such as m6A or CpG methylation) as
    small vertical rectangles at each modified position along the fibers.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing modification data.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw on. If None, a new axis will be created.
        The default is None.
    mod : str, optional
        The name of the modification layer to draw (e.g., 'm6a', 'cpg').
        The default is 'm6a'.
    width : float, optional
        The width (height) of each modification mark in axis units. The default is DEFAULT_WIDTH (0.8).
    color : str, optional
        The color of the modification marks. The default is '#000000' (black).
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with modifications drawn as vertical marks.
    """
    if ax == None:
        ax = make_plot_ax(fview)
    patch_list = []
    I, J = np.nonzero(fview.layers[mod])
    J_pos = [fview.var.pos[j] for j in J]
    for k in range(len(I)):
        patch = patches.Rectangle((J_pos[k], I[k] - 0.5 * width), lw=0,
                                  width=1, height=width, color=color, zorder=4)
        patch_list.append(patch)
    patch_coll = PatchCollection(patch_list, match_original=True)
    ax.add_collection(patch_coll)
    return(ax)


def draw_mods_offset(fview, ax=None, mod='m6a', width=DEFAULT_WIDTH, color='#000000'):
    # Not working. don't know the right way to transform, would be much faster though.
    if ax == None:
        ax = make_plot_ax(fview)
    # patch_list = []
    I, J = np.nonzero(fview.layers[mod])
    J_pos = [fview.var.pos[j] for j in J]
    patch = patches.Rectangle((0, - 0.5 * width), lw=0,
                                 width=1, height=width, color=color, zorder=4) 
    patch = patches.Rectangle((0, 0), lw=0,
                                 width=1, height=width, color=color, zorder=4) 
    patch_coll = PatchCollection([patch], match_original=True, 
                                 offsets=np.c_[J_pos, I], offset_transform=ax.transData)
    # patch_coll.set_offset_transform(ax.transData)
    patch_coll.set_transform(ax.transAxes)
    ax.add_collection(patch_coll)
    return(ax)


def draw_split_lines(fview, ax=None, split_var="site_name", color="black"):
    """
    Draw horizontal lines separating groups in a fiber view.
    
    This function draws horizontal dividing lines between groups of fibers based on
    a grouping variable in the observation metadata. This is useful for visually
    separating fibers from different sites or conditions.
    
    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object with grouped observations.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw on. If None, a new axis will be created.
        The default is None.
    split_var : str, optional
        The column name in obs to use for determining group boundaries.
        The default is "site_name".
    color : str, optional
        The color of the dividing lines. The default is "black".
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with horizontal dividing lines drawn between groups.
    """
    if ax == None:
        ax = make_plot_ax(fview)
    h_lines = []
    for i, group in enumerate(fview.obs[split_var]):
        if group != fview.obs[split_var][max(i-1, 0)]:
            h_lines.append(i-0.5)
    ax.hlines(h_lines, xmin=fview.var.pos[0], xmax=fview.var.pos[-1], color=color)
    return(ax)




# -----------------------------------------------------------------------------
# canned plotting functions


