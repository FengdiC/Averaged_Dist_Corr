import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
granddir = os.path.dirname(parentdir)
sys.path.insert(0, granddir)
from Components import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def load_hyperparam():
    logger.configure('./cartpole_tune/',['csv'], log_suffix='cartpole-summary-weighted-tune')
    for i in range(1,301,1):
        file = './cartpole_tune/progresscartpole_tune_shared-'+str(i)+'.csv'
        data = pd.read_csv(file, header=0,index_col='hyperparam')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        data = data.iloc[:, :250]
        results = []
        # if data.shape[0]<20:
        #     continue
        for j in range(30):
            rets = data.iloc[j,:].to_numpy()
            rets = np.squeeze(rets)
            results.append(rets)
        results= np.array(results)
        # for i in reversed(range(10, results.shape[1], 1)):
        #     results[:, i] = np.mean(results[:, i - 10:i + 1], axis=1)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(5)
        logger.logkv("hyperparam", list(data.iloc[:,1].index)[0]+'-'+str(i))
        for n in range(mean.shape[0]):
            logger.logkv(str(n), mean[n])
        logger.dumpkvs()
        # logger.logkv("hyperparam", str(data.iloc[:,1].index)+'--std')
        # columns = data.columns
        # for n in range(mean.shape[0]):
        #     logger.logkv(str(columns[n]), std[n])
    return -1

def compute_best():
    data = pd.read_csv('./cartpole_tune/progresscartpole-summary-weighted-tune.csv', header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    data = data.iloc[:, :250]
    best = data.max(axis=1)
    top_ten = best.nlargest(5)
    top_ten = top_ten.index
    results = data.loc[top_ten].to_numpy()
    top_ten = list(top_ten)
    plt.figure()
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]),results[i,:],label=top_ten[i])
    plt.legend()
    plt.show()
    return -1

def compute_final():
    data = pd.read_csv('./ppo/progressHopper-summary-naive-ppo-tune.csv', header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    data = data.iloc[:, :250]
    best = data.iloc[:,249]
    top_ten = best.nlargest(10)
    top_ten = top_ten.index
    results = data.loc[top_ten].to_numpy()
    top_ten = list(top_ten)
    plt.figure()
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]), results[i, :], label=top_ten[i])
    plt.legend()
    plt.title('final')
    plt.show()
    return -1

def plot_results(env):
    plt.figure()
    naive = []
    weighted = []
    biased_data = pd.read_csv('../Logs/PPO/mujoco-PPO-Ho-An-Ha.csv', header=0, index_col='hyperparam')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    biased = biased_data.loc['biased-'+env+'-mean']
    plt.plot(biased_data.columns,biased, color='blueviolet',label='biased')
    for seed in range(15):
        weighted_data = pd.read_csv('./ppo_run_2/progressHopper-weighted-ppo-tune-'+str(env)+'.csv',
                                    header=0,
                           parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        weighted_data.columns = weighted_data.columns.astype(int)
        weighted_data = weighted_data.sort_index(axis=1, ascending=True)
        # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
        # weighted.append(rets)
    weighted = weighted_data.to_numpy()
    # for i in reversed(range(10, weighted.shape[1], 1)):
    #     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
    mean = np.mean(weighted, axis=0)
    std = np.std(weighted, axis=0) / np.sqrt(10)
    plt.plot(weighted_data.columns,mean,color='orangered',label='our correction')

    for seed in range(15):
        naive_data = pd.read_csv('./ppo_run_2/progressHopper-naive-ppo-tune-'+str(env)+'.csv',
                                    header=0,
                           parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        naive_data.columns = naive_data.columns.astype(int)
        naive_data = naive_data.sort_index(axis=1, ascending=True)
        # rets = naive_data.loc[env + '-'+str(seed)].to_numpy()
        # naive.append(rets)
    naive = naive_data.to_numpy()
    # for i in reversed(range(10, naive.shape[1], 1)):
    #     naive[:, i] = np.mean(naive[:, i - 10:i + 1], axis=1)
    mean = np.mean(naive, axis=0)
    std = np.std(naive, axis=0) / np.sqrt(10)
    plt.plot(naive_data.columns,mean,color='dodgerblue',label='existing correction')
    plt.legend()
    plt.show()

# dummy_file()
load_hyperparam()
compute_best()
# compute_final()
# param = {'env': ['Hopper-v4', 'Swimmer-v4', 'Ant-v4']}
# plot_results('Swimmer-v4')