# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:49:12 2016

@author: salmon
"""

from functools import partial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # for plots
from matplotlib import rc
from prox_collection import l22_prox, l1_prox, l0_prox, scad_prox, mcp_prox, \
    log_prox, sqrt_prox, enet_prox
from prox_collection import l22_pen, l1_pen, l0_pen, \
    scad_pen, mcp_pen, log_pen, sqrt_pen, enet_pen

###############################################################################
# Plot initialization

plt.close('all')
dirname = "../srcimages/"
imageformat = '.pdf'


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'font.size': 16,
          'legend.fontsize': 16,
          'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
sns.axes_style()


def my_saving_display(fig, dirname, filename, imageformat):
    """"saving faster"""
    dirname + filename + imageformat
    image_name = dirname + filename + imageformat
    fig.savefig(image_name)

###############################################################################
# ploting prox operators and penalty functions


def plot_prox(x, threshold, prox, label, image_name):
    """Function to plot and save prox operators."""
    z = np.zeros(x.shape)
    for i, value in enumerate(np.nditer(x)):
        z[i] = prox(value, threshold)

    fig0 = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111)
    ax1.plot(x, z, label=label)
    # ax1.plot(x, log_prox(x, 3, 1),
    #          label=r"$\eta_{\rm {log},\lambda,\gamma}$")
    ax1.plot(x, x, 'k--', linewidth=1)
    plt.legend(loc="upper left", fontsize=34)
    ax1.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    plt.show()
    my_saving_display(fig0, dirname, image_name, imageformat)
    return fig0


def plot_pen(x, threshold, pen, image_name):
    """ function to plot and save pen functions"""
    xx = pen(x, threshold)
    fig0 = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111)
    ax1.plot(x, xx, label=label)
    # plt.title(label, fontsize=24)
    # plt.subplots_adjust(top=0.80)
    ax1.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.set_ylim(-0.1, np.max(xx) * 1.05)
    ax1.set_xlim(-10, 10)
    plt.show()
    my_saving_display(fig0, dirname, image_name + "pen", imageformat)
    return fig0


x = np.arange(-10, 10, step=0.01)


# no penalty
prox = l1_prox
image_name = "no_pen_orth_1d"
label = r"$\eta_{0}$"
plot_prox(x, 0, prox, label, image_name)
pen_l1 = l1_pen
plot_pen(x, 0, pen_l1, image_name)


# log prox
threshold = 4.5
epsilon = .5
label = r"$\eta_{\rm {log},\lambda,\gamma}$"
image_name = "log_orth_1d"
prox = partial(log_prox, epsilon=epsilon)
plot_prox(x, threshold, prox, label, image_name)
pen_log = partial(log_pen, epsilon=epsilon)
plot_pen(x, threshold, pen_log, image_name)

# mcp prox
threshold = 3
gamma = 2.5
label = r"$\eta_{\rm {MCP},\lambda,\gamma}$"
image_name = "mcp_orth_1d"
prox = partial(mcp_prox, gamma=gamma)
plot_prox(x, threshold, prox, label, image_name)
pen_mcp = partial(mcp_pen, gamma=gamma)
plot_pen(x, threshold, pen_mcp, image_name)

# SCAD prox
label = r"$\eta_{\rm {SCAD},\lambda,\gamma}$"
image_name = "scad_orth_1d"
prox = partial(scad_prox, gamma=gamma)
plot_prox(x, threshold, prox, label, image_name)
pen_scad = partial(scad_pen, gamma=gamma)
plot_pen(x, threshold, pen_scad, image_name)

# l1 prox
prox = l1_prox
image_name = "l1_orth_1d"
label = r"$\eta_{\rm {ST},\lambda}$"
plot_prox(x, threshold, prox, label, image_name)
pen_l1 = l1_pen
plot_pen(x, threshold, pen_l1, image_name)

# l22 prox
prox = l22_prox
label = r"$\eta_{\rm {Ridge},\lambda}$"
image_name = "l22_orth_1d"
plot_prox(x, threshold, prox, label, image_name)
pen_l22 = l22_pen
plot_pen(x, threshold, pen_l22, image_name)

# enet prox
beta = 1
label = r"$\eta_{\rm {Enet},\lambda,\gamma}$"
image_name = "enet_orth_1d"
prox = partial(enet_prox, beta=beta)
plot_prox(x, threshold, prox, label, image_name)
pen_enet = partial(enet_pen, beta=beta)
plot_pen(x, threshold, pen_enet, image_name)

# sqrt prox
label = r"$\eta_{\rm {sqrt},\lambda}$"
image_name = "sqrt_orth_1d"
prox = sqrt_prox
plot_prox(x, threshold, prox, label, image_name)
pen_sqrt = sqrt_pen
plot_pen(x, threshold, pen_sqrt, image_name)

# l0 prox
threshold = 4.5
label = r"$\eta_{\rm {HT},\lambda}$"
image_name = "l0_orth_1d"
prox = l0_prox
plot_prox(x, threshold, prox, label, image_name)
pen_l0 = l0_pen
plot_pen(x, threshold, pen_l0, image_name)

###############################################################################
# ploting penalty functions altogether

fig0 = plt.figure(figsize=(6, 6))
ax1 = plt.subplot(111)

ax1.plot(x, pen_l0(x, threshold), label='l0')
ax1.plot(x, pen_sqrt(x, threshold), label='sqrt')
ax1.plot(x, pen_l22(x, threshold), label='l22')
ax1.plot(x, pen_enet(x, threshold), label='enet')
ax1.plot(x, pen_log(x, threshold), label='log')
ax1.plot(x, pen_mcp(x, threshold), label='mcp')
ax1.plot(x, pen_scad(x, threshold), label='scad')
ax1.plot(x, pen_l1(x, threshold), label='l1')

ax1.set_ylim(-0.1, 40)
ax1.set_xlim(-10, 10)
ax1.get_yaxis().set_ticks([])
ax1.get_xaxis().set_ticks([])
plt.legend(loc="upper center", fontsize=14)

plt.show()
my_saving_display(fig0, dirname, "penalties", imageformat)
