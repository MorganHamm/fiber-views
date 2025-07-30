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
    s_pos = []
    e_pos = []
    mask = fview.layers['seq'] != b'-'
    for i in range(fview.shape[0]):
        s_pos.append(np.min(fview.var.pos[mask[i,:]]))
        e_pos.append(np.max(fview.var.pos[mask[i,:]]) + 1)
    fview.obs['s_pos'] = s_pos
    fview.obs['e_pos'] = e_pos
    return(None)


def make_plot_ax(fview):
    fig, ax = plt.subplots()
    ax.set_xlim(fview.var.pos[0], fview.var.pos[-1])
    ax.set_ylim(0, fview.shape[0])
    return(ax)


# -----------------------------------------------------------------------------
# plotting primitives

def draw_fiber_lines(fview, ax=None, color="#606060"):
    # good for fewer than ~150 fibers
    if ax == None:
        ax = make_plot_ax(fview)
    if ('s_pos' not in fview.obs.columns) | ('e_pos' not in fview.obs.columns):
        annotate_boundaries(fview)
    for i in range(fview.shape[0]):
        ax.hlines(y=i, xmin=fview.obs.s_pos.iloc[i], xmax=fview.obs.e_pos.iloc[i],
                 color=color, lw=0.5, zorder=1)
    return(ax)


def draw_fiber_bars(fview, ax=None, color="#d0d0d0", width=DEFAULT_WIDTH):
    # good for fewer than ~150 fibers
    if ax == None:
        ax = make_plot_ax(fview)
    if ('s_pos' not in fview.obs.columns) | ('e_pos' not in fview.obs.columns):
        annotate_boundaries(fview)
    patch_list = []
    for i in range(fview.shape[0]):
        patch = patches.Rectangle((fview.obs.s_pos.iloc[i], i - 0.5 * width), 
                                  width=fview.obs.e_pos.iloc[i] - fview.obs.s_pos.iloc[i], 
                                  height=width, color=color, zorder=1)
        patch_list.append(patch)
        # ax.add_patch(patch)
    patch_coll = PatchCollection(patch_list, match_original=True)
    ax.add_collection(patch_coll)
    return(ax)


def draw_regions(fview, ax=None, base_name='msp', color="red", width=DEFAULT_WIDTH):
    if ax == None:
        ax = make_plot_ax(fview)
    region_df = fv.tools.make_region_df(fview, base_name=base_name, zero_pos='center')
    patch_list = []
    for i, region in region_df.iterrows():
        patch = patches.Rectangle((region.start, region.row - 0.5 * width), 
                                  width=region.length, height=width, color=color, zorder=3)
        patch_list.append(patch)
    patch_coll = PatchCollection(patch_list, match_original=True)
    ax.add_collection(patch_coll)
    return(ax)


def draw_mods(fview, ax=None, mod='m6a', width=DEFAULT_WIDTH, color='#000000'):
    if ax == None:
        ax = make_plot_ax(fview)
    patch_list = []
    I, J = np.nonzero(fview.layers[mod])
    J_pos = [fview.var.pos[j] for j in J]
    for k in range(len(I)):
        patch = patches.Rectangle((J_pos[k], I[k] - 0.5 * width), 
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
    patch = patches.Rectangle((0, - 0.5 * width), 
                                 width=1, height=width, color=color, zorder=4) 
    patch = patches.Rectangle((0, 0), 
                                 width=1, height=width, color=color, zorder=4) 
    patch_coll = PatchCollection([patch], match_original=True, 
                                 offsets=np.c_[J_pos, I], offset_transform=ax.transData)
    # patch_coll.set_offset_transform(ax.transData)
    patch_coll.set_transform(ax.transAxes)
    ax.add_collection(patch_coll)
    return(ax)


def draw_split_lines(fview, ax=None, split_var="site_name", color="black"):
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


