import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/progressHopper-weighted-ppo.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:150]


param = {'lr': [0.0003],'lr_weight':[0.03,0.003,0.0003,0.0001],
         'scale_weight': [40]}

seeds = range(5)
steps = list(data)
plt.figure()
for values in list(itertools.product(param['lr'],param['lr_weight'],param['scale_weight'])):
    results = []

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
    std = np.std(results,axis=0)
    # plt.plot(steps, mean, color=color, label=line_name)
    # plt.plot(steps,mean, label=name)
    # plt.fill_between(steps, mean+std, mean-std,color=color, label=line_name,alpha=0.2,linewidth=0.9)
    # plt.errorbar(steps, mean, std, color=color, label=line_name, alpha=0.5, elinewidth=0.9)
    plt.errorbar(steps, mean, std,  alpha=0.5,label=name, elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
# plt.title('Ratio between our approximation bias and the wrong state distribution bias')
# set legend
plt.legend()
plt.show()