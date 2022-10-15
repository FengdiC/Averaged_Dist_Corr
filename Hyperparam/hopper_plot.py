import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('./spinningup/data/progressAnt-weighted-ppo-tune-gamma-coef.csv', header=0, index_col='hyperparam')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:,:250]

param = {'scale':[40,60],'gamma_coef':[3,1], 'target_kl':[0.01]}
# param = {'target_kl':[0.06,0.03,0.01],'pi_lr':[6e-4,3e-4,1e-4]}

seeds = range(3)
steps = list(data)
plt.figure()
for values in list(itertools.product(param['scale'], param['gamma_coef'],param['target_kl'])):
    results = []
    if values[0]==60 and values[1]==3:
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
    for i in reversed(range(10,results.shape[1],1)):
        results[:,i] = np.mean(results[:,i-10:i+1],axis=1)
    mean = np.mean(results, axis=0)
    std = 0.5*np.std(results,axis=0)
    # plt.plot(steps, mean, color=color, label=line_name)
    plt.plot(steps,mean, label=name)
    # plt.fill_between(steps, mean+std, mean-std,color=color, alpha=0.2,linewidth=0.9)
    plt.fill_between(steps, mean + std, mean - std, alpha=0.2, linewidth=0.9)
# plt.plot(steps,500 * np.ones(len(steps)),'--') #1/(1-gamma)

# define y_axis, x_axis
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
# plt.title('Ratio between our approximation bias and the wrong state distribution bias')
# set legend
plt.legend()
plt.show()