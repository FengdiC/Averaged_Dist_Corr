import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt


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
    ax.axes.set_ylim(0,None)
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

data = pd.read_csv('./logs/Reacher_all.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:20]

checkpoint = 1000

# name = ['ReLU-5-0.001-weighted_batch_ac','sigmoid-5-0.001-weighted_batch_ac','tanh-5-0.001-weighted_batch_ac',
#         'ReLU-5-0.0001-batch_ac_shared_gc','sigmoid-5-0.0001-batch_ac_shared_gc',
#         'tanh-5-0.001-batch_ac_shared_gc']
#
name = ['weighted_batch_ac-ReLU', 'batch_ac_shared_gc-ReLU']

seeds = range(30)
steps = list(data)
plt.figure(figsize=(11,7),dpi=110)
setsizes()
setaxes()
for values in list(itertools.product(name)):
    results = {'errs':[],'err-ratios':[],'errs-buffer':[],'rets':[]}
    if values[0] == "batch_ac_shared_gc-ReLU":
        line_name='shared'
        color = 'orangered'
    elif values[0] == "weighted_batch_ac-ReLU":
        line_name = 'non-shared'
        color = 'dodgerblue'
    else:
        line_name = 'tanh'
        color='blueviolet'

    for seed in seeds:
        name = [str(k) for k in values]
        name.append(str(seed))
        for key in results.keys():
            hyper_name = '-'.join(name) +'-'+key
            rets = data.loc[hyper_name].to_numpy()
            rets = np.squeeze(rets)
            # for i in range(rets.shape[0]):
            #     rets[i] = (1-gamma**rets[i])/(1-gamma)
            results[key].append(rets)

    plt.subplot(121)
    M = np.array(results['err-ratios'])
    mean = np.mean(M, axis=0)
    std = np.std(M,axis=0)/np.sqrt(30)
    plt.plot(steps, mean, color=color, label=line_name)
    plt.fill_between(steps,mean-std,mean+std,color=color,alpha=0.2)
    # plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)
    # plt.errorbar(steps, mean, std,  label=name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.yticks(fontsize=17)
    plt.xlabel("steps",fontsize=19)
    plt.ylabel("Ratio between biases",fontsize=19)
    plt.xticks(fontsize=17, rotation=45)
    plt.ylim(0, None)
    setaxes()
    # plt.title('Ratio between our approximation bias and the wrong state distribution bias')
    # plt.subplot(222)
    # M = np.array(results['errs'])
    # mean = np.mean(M, axis=0)
    # std = np.std(M, axis=0)
    # # plt.plot(steps, mean, color=color, label=line_name)
    # # plt.plot(steps,mean, label=name)
    # # plt.errorbar(steps, mean, std, color=color, =line_name, alpha=0.5, elinewidth=0.9)
    # plt.errorbar(steps, mean, std, label=name, alpha=0.5, elinewidth=0.9)
    # # define y_axis, x_axis
    # plt.xlabel("steps")
    # plt.ylabel("Averaged Errors")
    # plt.title('Errors between our approximation and the true correction averaged by the stationary distribuion')
    # plt.subplot(223)
    # M = np.array(results['errs-buffer'])
    # mean = np.mean(M, axis=0)
    # std = np.std(M, axis=0)
    # # plt.plot(steps, mean, color=color, label=line_name)
    # # plt.plot(steps,mean, label=name)
    # # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # plt.errorbar(steps, mean, std, label=name, alpha=0.5, elinewidth=0.9)
    # # define y_axis, x_axis
    # plt.xlabel("steps")
    # plt.ylabel("Averaged Errors")
    # plt.title('Errors between our approximation and the true correction averaged over the buffer')
    # plt.subplot(224)
    # M = np.array(results['rets'])
    # mean = np.mean(M, axis=0)
    # std = np.std(M, axis=0)
    # # plt.plot(steps, mean, color=color, label=line_name)
    # # plt.plot(steps,mean, label=name)
    # # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # plt.errorbar(steps, mean, std, label=name, alpha=0.5, elinewidth=0.9)
    # # define y_axis, x_axis
    # plt.xlabel("steps")
    # plt.ylabel("Undiscounted Returns")
    # plt.title('Returns')
plt.legend(prop={"size":17})


name = ['batch_ac_shared_gc-sigmoid','batch_ac_shared_gc-ReLU','batch_ac_shared_gc-tanh']
seeds = range(30)
steps = list(data)
for values in list(itertools.product(name)):
    results = {'errs':[],'err-ratios':[],'errs-buffer':[],'rets':[]}
    if values[0] == "batch_ac_shared_gc-ReLU":
        line_name='ReLU'
        color = 'orangered'
    elif values[0] == "batch_ac_shared_gc-sigmoid":
        line_name = 'sigmoid'
        color = 'dodgerblue'
    else:
        line_name = 'tanh'
        color='blueviolet'

    for seed in seeds:
        name = [str(k) for k in values]
        name.append(str(seed))
        for key in results.keys():
            hyper_name = '-'.join(name) +'-'+key
            rets = data.loc[hyper_name].to_numpy()
            rets = np.squeeze(rets)
            # for i in range(rets.shape[0]):
            #     rets[i] = (1-gamma**rets[i])/(1-gamma)
            results[key].append(rets)

    plt.subplot(122)
    M = np.array(results['err-ratios'])
    mean = np.mean(M, axis=0)
    std = np.std(M,axis=0)/np.sqrt(30)
    plt.plot(steps, mean, color=color, label=line_name)
    plt.fill_between(steps,mean - std, mean + std, color=color, alpha=0.2)
    # plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)
    # plt.errorbar(steps, mean, std,  label=name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.xticks(fontsize=17, rotation=45)
    plt.yticks(fontsize=17)
    plt.xlabel("steps",fontsize=19)
    plt.ylabel("Ratio between biases",fontsize=19)
    plt.ylim(0, None)
    setaxes()
    # plt.title('Ratio between our approximation bias and the wrong state distribution bias')
    # plt.subplot(222)
    # M = np.array(results['errs'])
    # mean = np.mean(M, axis=0)
    # std = np.std(M, axis=0)
    # # plt.plot(steps, mean, color=color, label=line_name)
    # # plt.plot(steps,mean, label=name)
    # # plt.errorbar(steps, mean, std, color=color, =line_name, alpha=0.5, elinewidth=0.9)
    # plt.errorbar(steps, mean, std, label=name, alpha=0.5, elinewidth=0.9)
    # # define y_axis, x_axis
    # plt.xlabel("steps")
    # plt.ylabel("Averaged Errors")
    # plt.title('Errors between our approximation and the true correction averaged by the stationary distribuion')
    # plt.subplot(223)
    # M = np.array(results['errs-buffer'])
    # mean = np.mean(M, axis=0)
    # std = np.std(M, axis=0)
    # # plt.plot(steps, mean, color=color, label=line_name)
    # # plt.plot(steps,mean, label=name)
    # # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # plt.errorbar(steps, mean, std, label=name, alpha=0.5, elinewidth=0.9)
    # # define y_axis, x_axis
    # plt.xlabel("steps")
    # plt.ylabel("Averaged Errors")
    # plt.title('Errors between our approximation and the true correction averaged over the buffer')
    # plt.subplot(224)
    # M = np.array(results['rets'])
    # mean = np.mean(M, axis=0)
    # std = np.std(M, axis=0)
    # # plt.plot(steps, mean, color=color, label=line_name)
    # # plt.plot(steps,mean, label=name)
    # # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # plt.errorbar(steps, mean, std, label=name, alpha=0.5, elinewidth=0.9)
    # # define y_axis, x_axis
    # plt.xlabel("steps")
    # plt.ylabel("Undiscounted Returns")
    # plt.title('Returns')

# set legend
plt.legend(prop={"size":17})
# plt.suptitle('Ratio between our approximation bias and the wrong state distribution bias',fontsize=18)
plt.tight_layout()
plt.show()

