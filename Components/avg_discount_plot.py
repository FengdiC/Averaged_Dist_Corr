import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/progress-Reacher_activation-hyperparam.csv', header=0, index_col='hyperparam-rets')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:20]
gamma = 0.99

param = {'agent':['weighted_batch_ac','batch_ac_shared_gc'],'lr_weight':[0.0001,0.0003,0.003,0.01],
         'closs_weight':[10,20]}
seeds = range(5)
steps = list(data)
plt.figure()
for values in list(itertools.product(['5-0.001-0.01-1-1-sigmoid','5-0.001-0.01-1-10-ReLU','5-0.001-0.01-1-1-tanh'])):
    lr_weight = values[0]
    # naive = bool(values[1])
    # weight_activation = values[1]
    # scale_weight= values[2]
    result = []

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
        name = '-'.join(name)
        rets = data.loc[name].to_numpy()
        rets = np.squeeze(rets)
        # for i in range(rets.shape[0]):
        #     rets[i] = (1-gamma**rets[i])/(1-gamma)
        result.append(rets)

    results = np.array(result)
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