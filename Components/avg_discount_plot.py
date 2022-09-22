import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/Reacher_shared.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:20]

name = ['ReLU-5-0.001-weighted_batch_ac','sigmoid-5-0.001-weighted_batch_ac','tanh-5-0.001-weighted_batch_ac',
        'ReLU-5-0.0001-batch_ac_shared_gc','sigmoid-5-0.0001-batch_ac_shared_gc',
        'tanh-5-0.001-batch_ac_shared_gc','batch_ac_shared_gc']

seeds = range(5)
steps = list(data)
plt.figure()
for values in list(itertools.product(name)):
    results = []

    # if agent == 'batch_ac' and epoch>1:
    #     continue
    # if agent == 'naive_batch_ac' and epoch>1:
    #     continue
    # if scale_weight>1 and weight_activation!='ReLU':
    #     continue

    # if weight_activation == "sigmoid":
    #     line_name='sigmoid'
    #     color = 'green'
    # elif weight_activation == "ReLU":
    #     line_name = 'ReLU'
    #     color = 'orange'
    # else:
    #     line_name = 'tanh'
    #     color='blue'

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
    # plt.plot(steps, mean, color=color, label=line_name)
    plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("undiscounted returns")
# set legend
plt.legend()
plt.show()