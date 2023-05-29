import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
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
def dummy_file():
    # for i in range(1,251,1):
    #     file = './dm_point_mass/progressbiased-ppo-tune-'+str(i)+'easy.csv'
    #     dummy = './dm_point_mass_ready/progressbiased-ppo-tune-'+str(i)+'easy.csv'
    for filename in os.listdir('better_task'):
        file = os.path.join('better_task_ready', filename)
        if not file.endswith('.csv'):
            continue
        # checking if it is a file
        dummy = os.path.join('better_task_ready', filename)
        print(file)
        # print(i)
        with open(file, 'r') as read_obj, open(dummy, 'w') as write_obj:
            # Iterate over the given list of strings and write them to dummy file as lines
            Lines = read_obj.readlines()
            Lines[0] = Lines[0].replace('\n',',hyperparam2\n')
            for line in Lines:
                write_obj.write(line)

def load_hyperparam():
    logger.configure('./dm_point_mass_ready/',['csv'], log_suffix='dm-point-mass-summary-biased-ppo-tune')
    for i in range(1,251,1):
        file = './dm_point_mass_ready/progressbiased-ppo-tune-'+str(i)+'easy.csv'
        data = pd.read_csv(file, header=0,
                           parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        data = data.iloc[:, :500]
        results = []
        if data.shape[0]<4:
            print(i)
            continue
        for j in range(data.shape[0]):
            rets = data.iloc[j,:].to_numpy()
            rets = np.squeeze(rets)
            results.append(rets.astype(float))
        results= np.array(results)
        # for i in reversed(range(10, results.shape[1], 1)):
        #     results[:, i] = np.mean(results[:, i - 10:i + 1], axis=1)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(data.shape[0])
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
    data = pd.read_csv('./dm_point_mass_ready/progressdm-point-mass-summary-biased-ppo-tune.csv', header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    data = data.iloc[:, :500]
    best = data.max(axis=1)
    top_ten = best.nlargest(5)
    top_ten = top_ten.index
    results = data.loc[top_ten].to_numpy()
    top_ten = list(top_ten)
    plt.figure()
    for i in range(results.shape[0]):
        plt.plot(range(results.shape[1]),results[i,:],label=top_ten[i])
    # num = [107,396,379]
    # for n in num:
    #     ant = data.iloc[n-1,:]
    #     plt.plot(range(results.shape[1]), ant.to_numpy(), label=ant.name)
    plt.legend()
    plt.show()
    return -1

def compute_final():
    data = pd.read_csv('./dm_point_mass_ready/progressdm-point-mass-summary-biased-ppo-tune.csv', header=0, index_col='hyperparam')
    data.columns = data.columns.astype(int)
    data = data.sort_index(axis=1, ascending=True)
    data = data.iloc[:, :500]
    best = data.iloc[:,490:500]
    best = best.mean(axis=1)
    top_ten = best.nlargest(8)
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

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.labelsize'] = 19.0
    plt.rcParams['axes.titlesize'] = 19.0
    plt.rcParams['xtick.labelsize'] = 16.0
    plt.rcParams['ytick.labelsize'] = 16.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0


def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=16, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=16, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(ax.getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(ax.getxticklabelsize())

def plot_new_results(env):
    plt.figure(figsize=(12, 6), dpi=60)
    plt.subplot(121)

    for seed in range(1):
        biased_data = pd.read_csv('./dm_point_mass_ready/progressbiased-ppo-tune-14easy.csv',
                                    header=0,
                           parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        biased_data.columns = biased_data.columns.astype(int)
        biased_data = biased_data.sort_index(axis=1, ascending=True)
        # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
        # weighted.append(rets)
    biased = biased_data.to_numpy().astype(float)
    # for i in reversed(range(10, weighted.shape[1], 1)):
    #     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
    mean = np.mean(biased, axis=0)
    std = np.std(biased, axis=0) / np.sqrt(10)
    print("biased: ",biased.shape[0])
    plt.plot(biased_data.columns,mean,color='tab:green',label='biased')
    plt.fill_between(biased_data.columns,mean +std, mean -std, color='tab:green', alpha=0.2, linewidth=0.9)

    for seed in range(1):
        weighted_data = pd.read_csv('./dm_point_mass_ready/progressweighted-ppo-tune-180easy.csv',
                                    header=0,
                           parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        weighted_data.columns = weighted_data.columns.astype(int)
        weighted_data = weighted_data.sort_index(axis=1, ascending=True)
        # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
        # weighted.append(rets)
    weighted = weighted_data.to_numpy()
    print("weighted: ",weighted.shape[0])
    # for i in reversed(range(10, weighted.shape[1], 1)):
    #     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
    weighted = weighted[:10].astype(float)
    mean = np.mean(weighted, axis=0)
    std = np.std(weighted, axis=0) / np.sqrt(10)
    plt.plot(weighted_data.columns,mean,color='tab:orange',label='our correction')
    plt.fill_between(biased_data.columns, mean +std, mean -std, color='tab:orange', alpha=0.2, linewidth=0.9)

    for seed in range(1):
        naive_data = pd.read_csv('./dm_point_mass_ready/progressnaive-ppo-tune-38easy.csv',
                                    header=0,
                           parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        naive_data.columns = naive_data.columns.astype(int)
        naive_data = naive_data.sort_index(axis=1, ascending=True)
        # rets = naive_data.loc[env + '-'+str(seed)].to_numpy()
        # naive.append(rets)
    naive = naive_data.to_numpy()
    print("naive: ",naive.shape[0])
    # for i in reversed(range(10, naive.shape[1], 1)):
    #     naive[:, i] = np.mean(naive[:, i - 10:i + 1], axis=1)
    mean = np.mean(naive, axis=0)
    std = np.std(naive, axis=0) / np.sqrt(10)
    plt.plot(naive_data.columns,mean,color='tab:blue',label='existing correction')
    plt.fill_between(biased_data.columns, mean +std, mean -std, color='tab:blue', alpha=0.2, linewidth=0.9)

    # define y_axis, x_axis
    plt.xlabel("steps",fontsize=16)
    plt.ylabel("Undiscounted Returns",fontsize=16)
    setaxes()
    # define y_axis, x_axis
    setsizes()
    plt.xticks(fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.title('Point Mass')

    plt.subplot(122)

    for seed in range(1):
        biased_data = pd.read_csv('./ppo_tune/mountain_ready/progressbiased-ppo-tune-196.csv',
                                  header=0,
                                  parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                                  index_col='timestamp')
        biased_data.columns = biased_data.columns.astype(int)
        biased_data = biased_data.sort_index(axis=1, ascending=True)
        # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
        # weighted.append(rets)
    biased = biased_data.to_numpy().astype(float)
    # for i in reversed(range(10, weighted.shape[1], 1)):
    #     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
    mean = np.mean(biased, axis=0)
    std = np.std(biased, axis=0) / np.sqrt(10)
    print("biased: ", biased.shape[0])
    plt.plot(biased_data.columns, mean, color='tab:green', label='biased')
    plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:green', alpha=0.2, linewidth=0.9)

    for seed in range(1):
        weighted_data = pd.read_csv('./ppo_tune/mountain_ready/progressweighted-ppo-tune-243.csv',
                                    header=0,
                                    parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                                    index_col='timestamp')
        weighted_data.columns = weighted_data.columns.astype(int)
        weighted_data = weighted_data.sort_index(axis=1, ascending=True)
        # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
        # weighted.append(rets)
    weighted = weighted_data.to_numpy()
    print("weighted: ", weighted.shape[0])
    # for i in reversed(range(10, weighted.shape[1], 1)):
    #     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
    weighted = weighted[:10].astype(float)
    mean = np.mean(weighted, axis=0)
    std = np.std(weighted, axis=0) / np.sqrt(10)
    plt.plot(weighted_data.columns, mean, color='tab:orange', label='our correction')
    plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

    for seed in range(1):
        naive_data = pd.read_csv('./ppo_tune/mountain_ready/progressnaive-ppo-tune-151.csv',
                                 header=0,
                                 parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                                 index_col='timestamp')
        naive_data.columns = naive_data.columns.astype(int)
        naive_data = naive_data.sort_index(axis=1, ascending=True)
        # rets = naive_data.loc[env + '-'+str(seed)].to_numpy()
        # naive.append(rets)
    naive = naive_data.to_numpy()
    print("naive: ", naive.shape[0])
    # for i in reversed(range(10, naive.shape[1], 1)):
    #     naive[:, i] = np.mean(naive[:, i - 10:i + 1], axis=1)
    mean = np.mean(naive, axis=0)
    std = np.std(naive, axis=0) / np.sqrt(10)
    plt.plot(naive_data.columns, mean, color='tab:blue', label='existing correction')
    plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:blue', alpha=0.2, linewidth=0.9)

    # define y_axis, x_axis
    plt.xlabel("steps", fontsize=16)
    plt.ylabel("Undiscounted Returns", fontsize=16)
    setaxes()
    # define y_axis, x_axis
    setsizes()
    plt.xticks(fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.title('Mountain Car Continuous')
    plt.tight_layout()
    plt.legend(prop={"size": 17})
    plt.show()

def plot_results(env):
    plt.figure()
    naive = []
    weighted = []
    # biased_data = pd.read_csv('../Logs/PPO/mujoco-PPO-Ho-An-Ha.csv', header=0, index_col='hyperparam')
    biased_data = pd.read_csv('../Logs/PPO/mujoco_simple.csv', header=0, index_col='hyperparam')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    biased = biased_data.loc['biased-'+env+'-mean']
    plt.plot(biased_data.columns,biased, color='blueviolet',label='biased')
    for seed in range(1):
        weighted_data = pd.read_csv('./mujoco/progressAnt-weighted-ppo-tune-'+str(env)+'.csv',
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

    for seed in range(1):
        naive_data = pd.read_csv('./mujoco/progressAnt-naive-ppo-tune-'+str(env)+'.csv',
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
# load_hyperparam()
# compute_best()
# compute_final()
# param = {'env': ['Hopper-v4', 'Swimmer-v4', 'Ant-v4']}
# plot_results('InvertedPendulum-v4')
plot_new_results('Ant-v4')