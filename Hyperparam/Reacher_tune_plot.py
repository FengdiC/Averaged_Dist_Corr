import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Components import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
def dummy_file():
    for i in range(1,501,1):
        file = './ppo_tune_results/progressHopper-weighted-ppo-tune-'+str(i)+'.csv'
        dummy = './ppo/progressHopper-weighted-ppo-tune-'+str(i)+'.csv'
        with open(file, 'r') as read_obj, open(dummy, 'w') as write_obj:
            # Iterate over the given list of strings and write them to dummy file as lines
            Lines = read_obj.readlines()
            Lines[0] = Lines[0].replace('\n',',hyperparam2\n')
            for line in Lines:
                write_obj.write(line)

def load_hyperparam():
    agent = 'batch_ac_shared_gc'
    naive = False
    print(agent,naive)
    logger.configure('./Reacher_repeated/', ['csv'], log_suffix='Reacher-summary-shared-ppo-tune')
    # logger.configure('./Reacher_repeated/',['csv'], log_suffix='Reacher-summary-weighted-ppo-tune')
    # logger.configure('./Reacher_repeated/', ['csv'], log_suffix='Reacher-summary-biased-ppo-tune')
    # logger.configure('./Reacher_repeated/', ['csv'], log_suffix='Reacher-summary-naive-ppo-tune')
    # param = {'agent': ['batch_ac_shared_gc', 'batch_ac', "weighted_batch_ac"], 'naive': [True, False]}
    for i in range(1,501,1):
        file = './Reacher_repeated/progressReacher_tune_no_repeat-'+str(i)+'.csv'
        data = pd.read_csv(file, header=0,index_col='hyperparam')
        hyper = data.index[30]
        data = pd.read_csv(file, header=0,
                           parse_dates={'timestamp': ['hyperparam','agent']},
                           index_col='timestamp')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        data = data.iloc[:, :100]
        name = [str(agent),str(naive)]
        name = '-'.join(name)
        name = hyper+' '+name
        mean = data.loc[name].to_numpy()
        logger.logkv("hyperparam", list(data.iloc[:,1].index)[0])
        for n in range(mean.shape[0]):
            logger.logkv(str(n), mean[n])
        logger.dumpkvs()
        # logger.logkv("hyperparam", str(data.iloc[:,1].index)+'--std')
        # columns = data.columns
        # for n in range(mean.shape[0]):
        #     logger.logkv(str(columns[n]), std[n])
    return -1

def compute_best():
    file = './Reacher_repeated/progressReacher-summary-naive-ppo-tune.csv'
    print(file)
    data = pd.read_csv(file, header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.dropna(axis=1, how='all')
    data = data.sort_index(axis=1, ascending=True)
    data = data.iloc[:, :100]
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

def compare_best():
    data = pd.read_csv('./Reacher_repeated/progressReacher-summary-naive-ppo-tune.csv', header=0,
                       index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    naive = data.loc['18.46-29-0.0046-32-5-0.0004-32-0.99-0 batch_ac_shared_gc-False-0'].to_numpy()
    data = pd.read_csv('./Reacher_repeated/progressReacher-summary-biased-ppo-tune.csv', header=0,
                       index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    biased = data.loc['16.14-76-0.0038-32-5-0.003-32-0.95-0 batch_ac_shared_gc-False-0'].to_numpy()
    data = pd.read_csv('./Reacher_repeated/progressReacher-summary-shared-ppo-tune.csv', header=0,
                       index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    shared = data.loc['11.65-6-0.0041-8-25-0.0036-64-0.95-0 batch_ac_shared_gc-False-0'].to_numpy()
    data = pd.read_csv('./Reacher_repeated/progressReacher-summary-weighted-ppo-tune.csv', header=0,
                       index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    data = data.iloc[:, :100]
    weighted = data.loc['19.08-9-0.003-32-5-0.0047-16-0.95-0 batch_ac_shared_gc-False-0'].to_numpy()
    plt.figure()
    plt.plot(naive,label='naive')
    plt.plot(biased, label='biased')
    plt.plot(shared, label='shared')
    plt.plot(weighted, label='weighted')
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
    for seed in range(10):
        weighted_data = pd.read_csv('../Logs/PPO_tuned/progressmujoco_ppo_weighted_simple='+str(seed)+'.csv',
                                    header=0, index_col='hyperparam')
        weighted_data.columns = weighted_data.columns.astype(int)
        weighted_data = weighted_data.sort_index(axis=1, ascending=True)
        rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
        weighted.append(rets)
    weighted = np.array(weighted)
    for i in reversed(range(10, weighted.shape[1], 1)):
        weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
    mean = np.mean(weighted, axis=0)
    std = np.std(weighted, axis=0) / np.sqrt(10)
    plt.plot(weighted_data.columns,mean,color='orangered',label='our correction')

    for seed in range(10):
        naive_data = pd.read_csv('../Logs/PPO_tuned/progressmujoco_ppo_naive_tuned='+str(seed)+'.csv',
                                    header=0, index_col='hyperparam')
        naive_data.columns = naive_data.columns.astype(int)
        naive_data = naive_data.sort_index(axis=1, ascending=True)
        rets = naive_data.loc[env + '-'+str(seed)].to_numpy()
        naive.append(rets)
    naive = np.array(naive)
    for i in reversed(range(10, naive.shape[1], 1)):
        naive[:, i] = np.mean(naive[:, i - 10:i + 1], axis=1)
    mean = np.mean(naive, axis=0)
    std = np.std(naive, axis=0) / np.sqrt(10)
    plt.plot(naive_data.columns,mean,color='dodgerblue',label='existing correction')
    plt.legend()
    plt.show()

# dummy_file()
# load_hyperparam()
compute_best()
# compare_best()
# param = {'env': ['Hopper-v4', 'Swimmer-v4', 'Ant-v4']}
# plot_results('Swimmer-v4')