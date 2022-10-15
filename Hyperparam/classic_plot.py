import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 17.0
    plt.rcParams['ytick.labelsize'] = 17.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0


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
                   labelsize=16, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=16, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())


data = pd.read_csv('./logs/classic-control/progressclassic-control.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:150]
# 10, 6
plt.figure(figsize=(22,6), dpi=70)
ax1 = plt.subplot(131)

param = {'agent': ['batch_ac_shared_gc', 'batch_ac'], 'naive': [True, False],
         'env': ['CartPole-v1']}

seeds = range(30)
steps = list(data)
for values in list(itertools.product(param['agent'],param['naive'],param['env'])):
    results = []

    if values[0] == 'batch_ac_shared_gc' and values[1] == True:
        continue
    elif values[0] == 'batch_ac_shared_gc':
        line_name = 'our correction'
        color = 'orangered'
    elif values[1] == True:
        line_name = 'existing corection'
        color = 'dodgerblue'
    else:
        line_name = 'biased'
        color = 'blueviolet'

    for seed in seeds:
        name = [str(k) for k in values]
        name.append(str(seed))
        name = '-'.join(name)
        rets = data.loc[name].to_numpy()
        rets = np.squeeze(rets)
        # for i in range(rets.shape[0]):
        #     rets[i] = (1-gamma**rets[i])/(1-gamma)
        results.append(rets)

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(30)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color,alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.xlabel("steps",fontsize=19)
plt.ylabel("Undiscounted Returns",fontsize=19)
ax1.xaxis.offsetText.set_fontsize(19)
setaxes()
plt.title('CartPole',fontsize=19)
# set legend
# plt.legend()

ax2=plt.subplot(132)

param = {'agent': ['batch_ac_shared_gc', 'batch_ac'], 'naive': [True, False],
         'env': ['Acrobot-v1']}

seeds = range(30)
steps = list(data)
for values in list(itertools.product(param['agent'],param['naive'],param['env'])):
    results = []

    if values[0] == 'batch_ac_shared_gc' and values[1] == True:
        continue
    elif values[0] == 'batch_ac_shared_gc':
        line_name = 'our correction'
        color = 'orangered'
    elif values[1] == True:
        line_name = 'existing corection'
        color = 'dodgerblue'
    else:
        line_name = 'biased'
        color = 'blueviolet'

    for seed in seeds:
        name = [str(k) for k in values]
        name.append(str(seed))
        name = '-'.join(name)
        rets = data.loc[name].to_numpy()
        rets = np.squeeze(rets)
        # for i in range(rets.shape[0]):
        #     rets[i] = (1-gamma**rets[i])/(1-gamma)
        results.append(rets)

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(30)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.xlabel("steps",fontsize=19)
plt.ylabel("Undiscounted Returns",fontsize=19)
ax1.xaxis.offsetText.set_fontsize(19)
setaxes()
plt.title('Repeated Acrobot',fontsize=19)
# set legend
# plt.legend()

ax3 = plt.subplot(133)

param = {'agent': ['batch_ac_shared_gc', 'batch_ac'], 'naive': [True, False],
         'env': ['MountainCarContinuous-v0']}

seeds = range(30)
steps = list(data)
for values in list(itertools.product(param['agent'],param['naive'],param['env'])):
    results = []

    if values[0] == 'batch_ac_shared_gc' and values[1] == True:
        continue
    elif values[0] == 'batch_ac_shared_gc':
        line_name = 'our correction'
        color = 'orangered'
    elif values[1] == True:
        line_name = 'existing corection'
        color = 'dodgerblue'
    else:
        line_name = 'biased'
        color = 'blueviolet'

    for seed in seeds:
        name = [str(k) for k in values]
        name.append(str(seed))
        name = '-'.join(name)
        rets = data.loc[name].to_numpy()
        rets = np.squeeze(rets)
        # for i in range(rets.shape[0]):
        #     rets[i] = (1-gamma**rets[i])/(1-gamma)
        results.append(rets)

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(30)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.xlabel("steps",fontsize=19)
plt.ylabel("Undiscounted Returns",fontsize=19)
ax1.xaxis.offsetText.set_fontsize(19)
setaxes()
plt.title('Repeated Mountain Car Continuous',fontsize=19)
# set legend
plt.tight_layout()
plt.legend(prop={"size":16})

plt.show()