import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from exe.config import *
import re
import dill
from collections import defaultdict
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

SPINE_COLOR = 'black'  # this was gray


def latexify(fig_width=None, fig_height=None, columns=1, labelsize=10):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    See https://nipunbatra.github.io/blog/2014/latexify.html for more detail.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    labelsize : int, optional
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    LABELSIZE = labelsize

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    params = {'backend': 'pdf',
              # 'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': LABELSIZE * 1.2,  # fontsize for x and y labels (was 10)
              'axes.titlesize': LABELSIZE * 1.2,
              'patch.linewidth': 0.1,
              'patch.edgecolor': 'white',
              'legend.fontsize': LABELSIZE,  # was 10
              'legend.loc': 'upper right',
              'legend.borderpad': 0.1,
              'xtick.labelsize': LABELSIZE,
              'ytick.labelsize': LABELSIZE,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def window(x, win=10):
    tmp = np.array(range(len(x)), dtype=float)
    counter = 0
    while counter < len(x):
        tmp[counter] = float(x[counter:counter + win].mean())
        if len(x[counter:]) < win:
            tmp[counter:] = float(x[counter:].mean())
        counter += 1
    return pd.Series(tmp)


def heatmap(data, filename, vmax, vmin, loc='lower left', bbox=(-0.1, 1.02, 1.1, 0.2)):
    latexify(10, labelsize=25)  # TODO: point
    fig, ax = plt.subplots()
    print(data)
    sns.heatmap(data=data, ax=ax, square=True, vmax=vmax, vmin=vmin, cmap='hot_r')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_ylim(len(data), 0)
    # plt.legend(title="heatmap", loc=loc, bbox_to_anchor=bbox)
    plt.tight_layout()
    format_axes(ax)  # TODO: point
    plt.savefig(filename)
    del fig
    del ax


def line_graph(data, x, y, hue, filename, loc='lower left', bbox=(-0.1, 1.02, 1.1, 0.2), mode='expand', ncol=5,
               palette_size=None):
    latexify(10, labelsize=20)  # TODO: point
    fig, ax = plt.subplots()
    if palette_size is None:
        sns.lineplot(x=x, y=y, hue=hue, data=data, ax=ax)
    else:
        palette = sns.color_palette("Set1", palette_size)
        sns.lineplot(x=x, y=y, hue=hue, palette=palette, data=data, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    plt.legend(handles=handles[1:], title=None, loc=loc, bbox_to_anchor=bbox, mode=mode, ncol=ncol)
    plt.tight_layout()
    format_axes(ax)  # TODO: point
    plt.savefig(filename)
    del fig
    del ax


def reward_plots(_mdp, _agent, _window=50):
    import glob
    csvfiles = []
    for a in _agent:
        csvfiles += [p for p in glob.glob(LOG_DIR + "{0}_{1}_*_fin.csv".format(a, _mdp))
                     if re.search(LOG_DIR + "{0}_{1}_[0-9][0-9]+_fin.csv".format(a, _mdp), p)]
    df = pd.read_csv(csvfiles[0])
    df["Cumulative Reward"] = df["Cumulative Reward"].rolling(window=_window).mean()
    print(csvfiles)
    for f in csvfiles[1:]:
        tmp = pd.read_csv(f)
        tmp["Cumulative Reward"] = tmp["Cumulative Reward"].rolling(window=_window).mean()
        df = df.append(tmp, ignore_index=True)

    data = df.loc[:, ["Episode", "Cumulative Reward", "AgentName", "seed"]]
    data.loc[data["AgentName"].isin(["QLearning", "qlearning"]), "AgentName"] = "Q-Learning"
    data.loc[data["AgentName"].isin(["DynaQ", "dynaq"]), "AgentName"] = "Dyna-Q"
    data.loc[data["AgentName"].isin(["darling", "Darling"]), "AgentName"] = "DARLING"
    print(data)
    line_graph(data, x="Episode", y="Cumulative Reward", hue="AgentName",
               filename=FIG_DIR + "REWARD_{0}.pdf".format(_mdp))

def parseOptions():
    import optparse
    optParser = optparse.OptionParser()
    optParser.add_option('-w', '--window', action='store',
                         type='int', dest='window', default=50,
                         help='Window size (default %default)')
    optParser.add_option('-a', '--agent', action='store', metavar="A",
                         type='string', dest='agent', default="Q-Learning",
                         help='Agent type '
                              '(options are \'q-learning\', \'sarsa\', \'rmax\', \'dynaq\' default %default)')
    optParser.add_option('--mdp', action='store', metavar="MDP",
                         type='string', dest='mdp', default="blockworld",
                         help='MDP name '
                              '(options are \'graphmap_uncertainty \'stationary \'gridworld\' default %default)')
    optParser.add_option('-c', '--compare', action='store', metavar="C",
                         type='string', dest='compare', default="methods",
                         help='compare option (options are \'variables\', \'methods\', and \'both\' default %default)')
    optParser.add_option('-x', '--xlabel', action='store', metavar="C",
                         type='string', dest='xlabel', default="gamma",
                         help='xlabel option '
                              '(options are \'gamma\', \'alpha\', \'epsilon\', \'rmax\', \'ucount\', and \'lookahead\' '
                              'default %default)')
    optParser.add_option('-p', '--prefix', action='store', metavar="P",
                         type='string', dest='prefix', default="",
                         help='filename prefix')

    return optParser.parse_args()


if __name__ == "__main__":
    opts, args = parseOptions()
    print(opts.mdp)
    if opts.compare == 'methods':
        print("Generating figures...")
        print(opts.agent.split(" "))
        reward_plots(opts.mdp, opts.agent.split(" "), _window=opts.window)
