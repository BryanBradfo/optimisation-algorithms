"""Created on Thu Oct  27 16:49:12 2016.

@author: salmon
"""

from functools import partial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # for plots
from prox_collection import l22_prox, l1_prox, l0_prox, scad_prox, mcp_prox, \
    log_prox, sqrt_prox, enet_prox, l22_objective, l1_objective, \
    l0_objective, scad_objective, mcp_objective, log_objective, \
    sqrt_objective, enet_objective
from matplotlib import rc
from matplotlib import animation
# from mpl_toolkits.axes_grid1 import host_subplot

np.random.seed(seed=44)

###############################################################################
# Plot initialization

plt.close('all')
dirname = "../srcimages/"
imageformat = '.pdf'


# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'font.size': 16,
          'legend.fontsize': 16,
          # 'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("ticks")
color_blind_list = sns.color_palette("colorblind", 8)
my_blue = color_blind_list[0]


def my_saving_display(fig, dirname, filename, imageformat):
    """"saving faster"""
    dirname + filename + imageformat
    image_name = dirname + filename + imageformat
    fig.savefig(image_name)


###############################################################################

# Uncomment the next two lines if you want to save the animation
# import matplotlib
# matplotlib.use("Agg")

threshold = 3.

# Choose among the following regularization:
regs = ['l22', 'l1', 'enet', 'l0', 'scad', 'mcp', 'log', 'sqrt']

for reg in regs:
    # Setup figure and subplots
    plt.close('all')
    f0 = plt.figure(num=0, figsize=(12, 4.5))  # dpi = 100)
    ax01 = plt.subplot2grid((1, 3), (0, 0))
    ax02 = plt.subplot2grid((1, 3), (0, 1))
    ax03 = plt.subplot2grid((1, 3), (0, 2))
    sns.despine(top=True, right=False, left=False, bottom=False)

    if reg == 'l22':
        prox = l22_prox
        objective = l22_objective
        title = 'l22_film.mp4'
        f0.suptitle("$|x|^2$")

    elif reg == 'l1':
        prox = l1_prox
        objective = l1_objective
        title = 'l1_film.mp4'
        f0.suptitle("l1")
        f0.suptitle("$|x|$")

    elif reg == 'enet':
        beta = float(.5)  # beta should be > 0
        prox = partial(enet_prox, beta=beta)
        objective = partial(enet_objective, beta=beta)
        title = 'enet_film.mp4'
        f0.suptitle("$|x|^2 + |x|$ (Elastic net)")

    elif reg == 'l0':
        threshold = threshold ** 2 / 2  # to get same threshold as l1
        prox = l0_prox
        objective = l0_objective
        title = 'l0_film.mp4'
        f0.suptitle("$1_{|x| > 0}$ ($\ell_0$)")

    elif reg == 'scad':
        gamma = float(2.5)  # gamma should be > 2
        prox = partial(scad_prox, gamma=gamma)
        objective = partial(scad_objective, gamma=gamma)
        title = 'scad_film.mp4'
        f0.suptitle("SCAD")

    elif reg == 'mcp':
        gamma = float(1.1)  # gamma should be > 1
        prox = partial(mcp_prox, gamma=gamma)
        objective = partial(mcp_objective, gamma=gamma)
        title = 'mcp_film.mp4'
        f0.suptitle("MCP")

    elif reg == 'log':
        threshold = float(3)   # to get same threshold as l1
        epsilon = threshold / 50.  # epsilon should be > 1
        prox = partial(log_prox, epsilon=epsilon)
        objective = partial(log_objective, epsilon=epsilon)
        title = 'log_film.mp4'
        f0.suptitle("$\log(|x| + \epsilon)$")

    elif reg == 'sqrt':
        prox = sqrt_prox
        objective = sqrt_objective
        title = 'sqrt_film.mp4'
        f0.suptitle("$\sqrt{|x|}$")

    plt.tight_layout()
    f0.subplots_adjust(top=0.75)

    # Set titles of subplots
    ax01.set_title('Function to optimize')
    ax02.set_title('Proximal operator')
    ax03.set_title('Moreau envelop')


    # Data Placeholders
    prox_val = np.zeros(0)
    huber_val = np.zeros(0)
    t = np.zeros(0)

    x = -10.
    x_tab = np.arange(-10, 10, step=0.05)
    x_min = np.argmin(l1_objective(x_tab, x, threshold))

    # set plots
    p011, = ax01.plot(t, prox_val, '-', c=my_blue)
    p012, = ax01.plot(x_tab[x_min], prox(x, threshold), 'o', c=my_blue)
    p021, = ax02.plot(t, prox_val, '-', c=my_blue)
    p031, = ax03.plot(t, huber_val, '-', c=my_blue)

    # set y-limits
    ax01.set_ylim(-2, 40)
    ax02.set_ylim(-10, 10)
    ax03.set_ylim(-2, 40)

    # sex x-limits
    ax01.set_xlim(-10, 10)
    ax02.set_xlim(-10, 10)
    ax03.set_xlim(-10, 10)
    ax02.set_aspect('equal', 'datalim')


    def update_data(self):
        """ updating step """
        global x
        global prox_val
        global huber_val
        global t

        tmpv1 = prox(x, threshold)
        prox_val = np.append(prox_val, tmpv1)
        huber_val = np.append(huber_val, objective(tmpv1, x, threshold))
        t = np.append(t, x)
        x += 0.05

        p011.set_data(x_tab, objective(x_tab, x, threshold))  # objective value
        p012.set_data(tmpv1, objective(tmpv1, x, threshold))  # point value
        p021.set_data(t, prox_val)
        p031.set_data(t, huber_val)

        return p011, p012, p031

    simulation = animation.FuncAnimation(f0, update_data, blit=False, frames=400,
                                         interval=1, repeat=False)

    # Uncomment the next line if you want to save the animation
    simulation.save(title, fps=30)

    plt.show()
