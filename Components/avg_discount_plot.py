import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./logs/CartPole_PPO_large_gamma.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:450]
gamma = 0.99

param = {'agent':['ppo','naive_ppo','weighted_ppo'],'epoch':[1,10]}
seeds = range(4)
steps = list(data)
plt.figure()
for values in list(itertools.product(param['agent'], param['epoch'])):
    agent = values[0]
    # naive = bool(values[1])
    epoch = values[1]
    result = []

    if agent == 'batch_ac' and epoch>1:
        continue
    if agent == 'naive_batch_ac' and epoch>1:
        continue
    if agent =="weighted_batch_ac" and epoch==1:
        continue
    elif agent == 'weighted_batch_ac':
        line_name='averaged correction'
        color = 'green'
    elif agent == 'naive_batch_ac':
        line_name = 'existing correction'
        color = 'orange'
    else:
        line_name = 'biased'
        color='blue'

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
    plt.errorbar(steps, mean, std,color=color, label=line_name,alpha=0.5,elinewidth=0.9)

plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("undiscounted returns")
# set legend
plt.legend()
plt.show()