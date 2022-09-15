import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/progressclassic-control-weighted-batch-ac.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:150]


param = {'agent': ['batch_ac_shared_gc', 'batch_ac'], 'naive': [True, False],
         'env': ['Acrobot-v1']}

seeds = range(15)
steps = list(data)
plt.figure()
for values in list(itertools.product(param['agent'],param['naive'],param['env'])):
    results = []
    # if values[0] == "ReLU-batch_ac_shared_gc":
    #     line_name='shared'
    #     color = 'orangered'
    # elif values[0] == "ReLU-weighted_batch_ac":
    #     line_name = 'non-shared'
    #     color = 'dodgerblue'
    # else:
    #     line_name = 'tanh'
    #     color='blueviolet'

    if values[0] == 'batch_ac_shared_gc' and values[1] == True:
        continue

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
    plt.plot(steps,mean, label=name)
    # plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)

# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Ratio between biases")
plt.title('Ratio between our approximation bias and the wrong state distribution bias')
# set legend
plt.legend()
plt.show()