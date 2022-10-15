import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.labelsize'] = 19.0
    plt.rcParams['axes.titlesize'] = 19.0
    plt.rcParams['xtick.labelsize'] = 16.0
    plt.rcParams['ytick.labelsize'] = 16.0
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
    #     tick.label.set_fontsize(ax.getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(ax.getxticklabelsize())

data = pd.read_csv('./logs/PPO/mujoco-PPO-Ho-An-Ha.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:500]
# 10, 6
plt.figure(figsize=(25,12), dpi=60)
setaxes()
setsizes()
plt.subplot(241)

param = {'agent': ['biased', 'naive','ours40'],'env': ['Hopper-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'ours40':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours60':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0) /np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color,alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('Hopper')
# set legend
# plt.legend()

plt.subplot(242)

param = {'agent': ['biased', 'naive','ours60'],'env': ['Ant-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'ours60':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours40':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0) /np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('Ant')
# set legend
# plt.legend()

plt.subplot(243)

param = {'agent': ['biased', 'naive','ours60'],'env': ['HalfCheetah-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'ours60':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours40':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('HalfCheetah')

plt.subplot(244)

param = {'agent': ['biased', 'naive','ours60'],'env': ['Swimmer-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'ours60':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours40':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('Swimmer')

plt.subplot(245)

param = {'agent': ['biased', 'naive','ours60'],'env': ['Walker2d-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'ours60':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours40':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('Walker2d')

data = pd.read_csv('./logs/PPO/mujoco_simple.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:250]
plt.subplot(246)

param = {'agent': ['biased', 'naive','corrected'],'env': ['InvertedPendulum-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'corrected':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours60':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color,alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('InvertedPendulum')
plt.subplot(247)

param = {'agent': ['biased', 'naive','corrected'],'env': ['InvertedDoublePendulum-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'],param['env'])):
    results = []

    if values[0] == 'corrected':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours60':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)/np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean+std, mean-std,color=color,alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('InvertedDoublePendulum')

plt.subplot(248)

param = {'agent': ['biased', 'naive', 'corrected'], 'env': ['Reacher-v4']}

seeds = range(10)
steps = list(data)
for values in list(itertools.product(param['agent'], param['env'])):
    results = []

    if values[0] == 'corrected':
        line_name = 'our correction'
        color = 'orangered'
    elif values[0] == 'naive':
        line_name = 'existing correction'
        color = 'dodgerblue'
    elif values[0] == 'biased':
        line_name = 'biased'
        color = 'blueviolet'
    elif values[0] == 'ours60':
        line_name = 'ours60'
        color = 'red'
    else:
        line_name = 'separate'
        color = 'green'

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
    for i in reversed(range(10, results.shape[1], 1)):
        results[:, i] = np.mean(results[:, i - 10:i + 1], axis=1)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)/np.sqrt(10)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.fill_between(steps, mean + std, mean - std, color=color, alpha=0.2, linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # set legend
    plt.legend(prop={"size":19})
    # plt.legend()
# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
plt.title('Reacher')
plt.tight_layout()
plt.show()