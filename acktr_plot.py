from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob

def human_format_numbers(num, use_float=False):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 14.0
    plt.rcParams['ytick.labelsize'] = 14.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 3.0
    plt.rcParams['ytick.minor.pad'] = 50.0

    # https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
    plt.rcParams['pdf.fonttype'] = 42   

def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=12, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=12, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    for tick in ax.xaxis.get_major_ticks():
        # tick.label.set_fontsize(getxticklabelsize())
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(getxticklabelsize())
        tick.label.set_fontsize(14)


def avg_run_plot():
    """
    Args:
        data: 2D array; Each row contains returns, equally spaced bins
    """

    colors = ["tab:blue", "tab:green", "tab:orange"]
    x_tick = 10000

    agents = ["weighted_acktr_False", "acktr_False", "weighted_acktr_True"]
    envs = ["CartPole-v1", "InvertedPendulum-v2"]
    labels = ["weighted_acktr", "acktr", "naive_acktr"]
    
    
    for env in envs:
        i = 0        
        for agent in agents:
            key = "./results/{}_{}*".format(env, agent)
            all_paths = glob.glob(key)

            data = []
            for fp in all_paths:
                data.append(np.loadtxt(fp))
            data = np.array(data)

            avg_rets = np.mean(data, axis=0)

            std_errs = np.std(data, axis=0) / (len(data) - 1)
            x = np.arange(1, len(avg_rets) + 1) * x_tick
            plt.plot(x, avg_rets, color=colors[i], linewidth=2, label=labels[i])
            plt.fill_between(x, avg_rets - std_errs, avg_rets + std_errs, color=colors[i], alpha=0.4)
            i += 1
        plt.xlabel('Time-steps')
        h = plt.ylabel("Return", labelpad=25)
        h.set_rotation(0)
        plt.legend()
        plt.tight_layout()
        plt.pause(0.001)            
        plt.show()       

if __name__ == "__main__":
    avg_run_plot()
  