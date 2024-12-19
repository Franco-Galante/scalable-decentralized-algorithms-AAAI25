# This plotting script assumes that 'collaborative_training.py' has been execu-
# ted and the accuracy results on the test have been downloaded from wandb as
# 'accuracy_local-1.csv', 'accuracy_nocut-1.csv', 'accuracy_model-1.csv'

from itertools import groupby
from operator import itemgetter
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set_theme(
    context="talk", 
    style="ticks", 
    palette="deep", 
    font="sans-serif",
    color_codes=True, 
    font_scale=0.9,
    rc={
        'figure.facecolor': 'white', 
        "font.family": "sans-serif", 
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'legend.fontsize':12,
        'lines.markersize': 4.5, 
        'lines.linewidth': 1.15,
        'lines.linestyle': '--',
        'lines.marker': 'o'
    }
)
# sns.set_style('whitegrid') # add a grid to the plots

f1 = 'accuracy_local-1.csv'
f2 = 'accuracy_nocut-1.csv'
f3 = 'accuracy_model-1.csv'

df_local = pd.read_csv(f1)
df_nocut = pd.read_csv(f2)
df_model = pd.read_csv(f3)

fig, ax1 = plt.subplots(figsize=(5.6,3.4))

# precision on the removed links plot

# in 'edge_rem_xx_run-xx.csv' the info is basically saved as a dictionary
# the keys are in the first row (the links) and the values are the second
# row, corresponding to the instant of removal

df_rem = pd.read_csv('edge_rem_run-1.csv', sep='\t')

dict_rem = df_rem.squeeze().to_dict()

N_NODES = 100     # harcoded for the experiment
N_COMMUNITIES = 2

n_wrong_links = (N_NODES // N_COMMUNITIES) ** 2
wrong_links = []
for i in range(N_NODES // N_COMMUNITIES):
    wrong_links.extend([(i,(N_NODES // N_COMMUNITIES)+j) for j in range((N_NODES // N_COMMUNITIES))])

assert len(wrong_links) == n_wrong_links

# sort dictionary by value and then group those with same value
dict_rem_new = {ast.literal_eval(k): v for k, v in dict_rem.items()}
sorted_dict = sorted(dict_rem_new.items(), key=lambda x: x[1])
grouped_dict = {k: [i[0] for i in g] for k, g in groupby(sorted_dict, key=itemgetter(1))} # k: time, v: list of links

rem_so_far = []
counts = []
for time_instant in range(max(grouped_dict.keys()) + 1):
    if time_instant in grouped_dict:
        rem_so_far.extend(grouped_dict[time_instant])
    count = sum(1 for key in rem_so_far if key in wrong_links)
    counts.append(count)

ax1.plot([time_instant for time_instant in range(max(grouped_dict.keys()) + 1)], 
         [(n_wrong_links-c)/n_wrong_links for c in counts],
         linestyle='dotted', marker='', linewidth=1.1, color='blue') # , label='Inter-Class Link Fraction'
ax1.set_ylim([0,1.0])
ax1.set_xlim([0, 30])
ax1.set_ylabel(r'$\text{Wrong Links}$')
ax1.set_xlabel(r'$\text{Communication Round}$')


ax1.grid(True, axis='x')

# accuracy plots 

ax2 = ax1.twinx() # same graph, same x-axes

ax1.yaxis.label.set_color('blue')
ax1.tick_params(axis='y', colors='blue')
ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax1.set_yticks([0.0, 0.5, 1.0])

ax2.plot([np.nan] + df_local['run-5 - LOCAL Accuracy'].tolist(), linewidth=1.8, marker='', color='tab:orange', label='Local')
ax2.plot([np.nan] + df_nocut['run-5 - Model-NO-CUT Accuracy'].tolist(), linewidth=1.8, marker='', color='tab:red', label='FL-SG')
ax2.plot([np.nan] + df_model['run-5 - ML-C-COLME Accuracy'].tolist(), linewidth=1.8, marker='', color='tab:green', label='FL-DG')
ax2.set_ylabel(r'$\text{Accuracy [%]}$') # , labelpad=12
ax2.set_ylim([0,100])

ax2.grid(True, axis='y')

ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
ax1.tick_params(bottom=False, left=False, right=False)  # remove the tick
ax2.tick_params(bottom=False, left=False, right=False) # remove the ticks

# sns.despine()  # Remove the top and right spines
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_color('#dddddd')
plt.tight_layout()
plt.show()
