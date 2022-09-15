import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/architecture/Reacher/Reacher_shared.csv', header=0, index_col='hyperparam')
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
    results = []
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
        name = '-'.join(name) +'-err-ratios'
        rets = data.loc[name].to_numpy()
        rets = np.squeeze(rets)
        # for i in range(rets.shape[0]):
        #     rets[i] = (1-gamma**rets[i])/(1-gamma)
        results.append(rets)

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results,axis=0)
    plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Ratio between biases")
plt.title('Ratio between our approximation bias and the wrong state distribution bias')
# set legend
plt.legend()
plt.show()