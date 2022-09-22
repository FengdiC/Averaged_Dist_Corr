import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/progress-Reacher_repeat_shared_network.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:20]


name = ['ReLU-5-0.001-weighted_batch_ac','sigmoid-5-0.001-weighted_batch_ac','tanh-5-0.001-weighted_batch_ac',
        'ReLU-5-0.0001-batch_ac_shared_gc','sigmoid-5-0.0001-batch_ac_shared_gc',
        'tanh-5-0.001-batch_ac_shared_gc']

name = ['ReLU-weighted_batch_ac', 'ReLU-batch_ac_shared_gc']

# for agent in ['batch_ac_shared_gc']:
#     for activation in ['ReLU','sigmoid','tanh']:
#         name.append(activation+'-'+agent)

# agent = ['batch_ac_shared_gc']
# activation = ['ReLU','sigmoid','tanh']

seeds = range(5)
steps = list(data)
plt.figure()
for values in list(itertools.product(name)):
    results = {'errs':[],'err-ratios':[],'errs-buffer':[],'rets':[]}
    if values[0] == "ReLU-batch_ac_shared_gc":
        line_name='shared'
        color = 'orangered'
    elif values[0] == "ReLU-weighted_batch_ac":
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

    plt.subplot(221)
    M = np.array(results['err-ratios'])
    mean = np.mean(M, axis=0)
    std = np.std(M,axis=0)
    # plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)
    # define y_axis, x_axis
    plt.xlabel("steps")
    plt.ylabel("Ratio between biases")
    plt.title('Ratio between our approximation bias and the wrong state distribution bias')
    plt.subplot(222)
    M = np.array(results['errs'])
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    # plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.xlabel("steps")
    plt.ylabel("Averaged Errors")
    plt.title('Errors between our approximation and the true correction averaged by the stationary distribuion')
    plt.subplot(223)
    M = np.array(results['errs-buffer'])
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    # plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.xlabel("steps")
    plt.ylabel("Averaged Errors")
    plt.title('Errors between our approximation and the true correction averaged over the buffer')
    plt.subplot(224)
    M = np.array(results['rets'])
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    # plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    # define y_axis, x_axis
    plt.xlabel("steps")
    plt.ylabel("Undiscounted Returns")
    plt.title('Returns')

# set legend
plt.legend()
plt.show()