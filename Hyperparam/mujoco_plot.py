import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

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

# 10, 6
plt.figure(figsize=(25,6), dpi=60)
setaxes()
setsizes()

plt.subplot(141)

for seed in range(1):
    biased_data = pd.read_csv('./ppo_tune/ant/progressbiased-ppo-tune-16.csv',
                              header=0,
                              parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                              index_col='timestamp')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
    # weighted.append(rets)
biased = biased_data.to_numpy()
# for i in reversed(range(10, weighted.shape[1], 1)):
#     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
mean = np.mean(biased, axis=0)
std = np.std(biased, axis=0) / np.sqrt(10)
print("biased: ", biased.shape[0])
plt.plot(biased_data.columns, mean, color='tab:green', label='biased')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:green', alpha=0.2, linewidth=0.9)

for seed in range(1):
    weighted_data = pd.read_csv('./ppo_tune/ant/progressweighted-ppo-tune-790.csv',
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
mean = np.mean(weighted, axis=0)
std = np.std(weighted, axis=0) / np.sqrt(10)
plt.plot(weighted_data.columns, mean, color='tab:orange', label='our correction')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

for seed in range(1):
    naive_data = pd.read_csv('./ppo_tune/ant/progressnaive-ppo-tune-790.csv',
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
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
# define y_axis, x_axis
setsizes()
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.title('Ant')

plt.subplot(142)

for seed in range(1):
    biased_data = pd.read_csv('./ppo_tune/halfcheetah/progressbiased-ppo-tune-280.csv',
                              header=0,
                              parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                              index_col='timestamp')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
    # weighted.append(rets)
biased = biased_data.to_numpy()
# for i in reversed(range(10, weighted.shape[1], 1)):
#     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
mean = np.mean(biased, axis=0)
std = np.std(biased, axis=0) / np.sqrt(10)
print("biased: ", biased.shape[0])
plt.plot(biased_data.columns, mean, color='tab:green', label='biased')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:green', alpha=0.2, linewidth=0.9)

for seed in range(1):
    weighted_data = pd.read_csv('./ppo_tune/halfcheetah/progressweighted-ppo-tune-556.csv',
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
mean = np.mean(weighted, axis=0)
std = np.std(weighted, axis=0) / np.sqrt(10)
plt.plot(weighted_data.columns, mean, color='tab:orange', label='our correction')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

for seed in range(1):
    naive_data = pd.read_csv('./ppo_tune/halfcheetah/progressnaive-ppo-tune-535.csv',
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
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
# define y_axis, x_axis
setsizes()
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.title('HalfCheetah')

plt.subplot(143)

for seed in range(1):
    biased_data = pd.read_csv('./ppo_tune/swimmer/progressbiased-ppo-tune-394.csv',
                              header=0,
                              parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                              index_col='timestamp')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
    # weighted.append(rets)
biased = biased_data.to_numpy()
# for i in reversed(range(10, weighted.shape[1], 1)):
#     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
mean = np.mean(biased, axis=0)
std = np.std(biased, axis=0) / np.sqrt(10)
print("biased: ", biased.shape[0])
plt.plot(biased_data.columns, mean, color='tab:green', label='biased')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:green', alpha=0.2, linewidth=0.9)

for seed in range(1):
    weighted_data = pd.read_csv('./ppo_tune/swimmer/progressweighted-ppo-tune-394.csv',
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
mean = np.mean(weighted, axis=0)
std = np.std(weighted, axis=0) / np.sqrt(10)
plt.plot(weighted_data.columns, mean, color='tab:orange', label='our correction')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

for seed in range(1):
    naive_data = pd.read_csv('./ppo_tune/swimmer/progressnaive-ppo-tune-750.csv',
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
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
# define y_axis, x_axis
setsizes()
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.title('Swimmer')

plt.subplot(144)

for seed in range(1):
    biased_data = pd.read_csv('./ppo_tune/walker/progressbiased-ppo-tune-39.csv',
                              header=0,
                              parse_dates={'timestamp': ['hyperparam', 'hyperparam2']},
                              index_col='timestamp')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
    # weighted.append(rets)
biased = biased_data.to_numpy()
# for i in reversed(range(10, weighted.shape[1], 1)):
#     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
mean = np.mean(biased, axis=0)
std = np.std(biased, axis=0) / np.sqrt(10)
print("biased: ", biased.shape[0])
plt.plot(biased_data.columns, mean, color='tab:green', label='biased')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:green', alpha=0.2, linewidth=0.9)

for seed in range(1):
    weighted_data = pd.read_csv('./ppo_tune/walker/progressweighted-ppo-tune-566.csv',
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
mean = np.mean(weighted, axis=0)
std = np.std(weighted, axis=0) / np.sqrt(10)
plt.plot(weighted_data.columns, mean, color='tab:orange', label='our correction')
plt.fill_between(biased_data.columns, mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

for seed in range(1):
    naive_data = pd.read_csv('./ppo_tune/walker/progressnaive-ppo-tune-750.csv',
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
plt.xlabel("steps")
plt.ylabel("Undiscounted Returns")
setaxes()
# define y_axis, x_axis
setsizes()
plt.xticks(fontsize=17, rotation=45)
plt.yticks(fontsize=17)
plt.legend(prop={"size":17})
plt.title('Walker')
plt.tight_layout()
plt.show()