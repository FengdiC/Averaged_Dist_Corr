import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/architecture/Reacher/Reacher_shared.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:20]

checkpoint = 1000

# name = ['ReLU-5-0.001-weighted_batch_ac','sigmoid-5-0.001-weighted_batch_ac','tanh-5-0.001-weighted_batch_ac',
#         'ReLU-5-0.0001-batch_ac_shared_gc','sigmoid-5-0.0001-batch_ac_shared_gc',
#         'tanh-5-0.001-batch_ac_shared_gc']
#
name = ['ReLU-weighted_batch_ac-0.005-0.01', 'batch_ac_shared_gc-0.0005-0.001-1-5']

seeds = range(5)
steps = list(data)
plt.figure(figsize=(12,8), dpi=80)
for values in list(itertools.product(name)):
    results = {'errs':[],'err-ratios':[],'errs-buffer':[],'rets':[]}
    if values[0] == "batch_ac_shared_gc-0.0005-0.001-1-5":
        line_name='shared'
        color = 'orangered'
    elif values[0] == "ReLU-weighted_batch_ac-0.005-0.01":
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
    std = np.std(M,axis=0)
    plt.plot(steps, mean, color=color, label=line_name)
    plt.fill_between(steps,mean-std,mean+std,color=color,alpha=0.5)
    # plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)
    # plt.errorbar(steps, mean, std,  label=name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=15)
    plt.xlabel("steps",fontsize=18)
    plt.ylabel("Ratio between biases",fontsize=18)
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
plt.legend(prop={"size":16})


name = ['batch_ac_shared_gc-sigmoid','batch_ac_shared_gc-0.0005-0.001-1-5','batch_ac_shared_gc-tanh']
seeds = range(5)
steps = list(data)
for values in list(itertools.product(name)):
    results = {'errs':[],'err-ratios':[],'errs-buffer':[],'rets':[]}
    if values[0] == "batch_ac_shared_gc-0.0005-0.001-1-5":
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
    std = np.std(M,axis=0)
    plt.plot(steps, mean, color=color, label=line_name)
    plt.fill_between(steps,mean - std, mean + std, color=color, alpha=0.5)
    # plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)
    # plt.errorbar(steps, mean, std,  label=name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=15)
    plt.xlabel("steps",fontsize=18)
    plt.ylabel("Ratio between biases",fontsize=18)
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
plt.legend(prop={"size":16})
# plt.suptitle('Ratio between our approximation bias and the wrong state distribution bias',fontsize=18)
plt.show()

