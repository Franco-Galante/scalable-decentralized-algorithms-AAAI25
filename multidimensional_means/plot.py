import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import scipy.stats as stats
import seaborn as sns
sns.set_theme(
    context="talk", 
    style="ticks", 
    palette="deep", 
    font="sans-serif",
    color_codes=True, 
    font_scale=1.1,
    rc={
        'figure.facecolor': 'white', 
        "font.family": "sans-serif", 
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'legend.fontsize':18,
        'lines.markersize': 4.5, 
        'lines.linewidth': 1.2,
        'lines.linestyle': '--',
        'lines.marker': ''
    }
)
color_list = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan' 
    ]
sns.set_style('whitegrid') # add a grid to the plots

# to be called within a subfolder experiment (subfolder of 'res')
# the name of the subfolder already specifies most experiment vars
# in each folder there are different sample paths (seed) and k values

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data, axis=0)
    stderr = stats.sem(data, axis=0)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, margin_of_error



if __name__ == "__main__":

    cwd = os.path.basename(os.getcwd())

    if not cwd.startswith('GN_'):
        raise RuntimeError('Call the script within an experiment folder')

    cwd_split = cwd.split('_')
    if len(cwd_split) != 7:
        raise RuntimeError('Folder with unrecognized specification')

    H = 7000 # HARDCODED (to be matched with the longest sequence)
    log_flag = True
    norm_flag = True
    plot_path = False
    file_path = 'ts_summary.csv' # out file

    N = cwd_split[1]
    r = cwd_split[2][1:]
    sigma = cwd_split[4][1:]

    epsilon = cwd_split[5][1:]
    delta = cwd_split[6][1:]

    exp = cwd_split[3][3:] # one of three possible types (mu of classes)

    # ----------------------- Raw data preprocessing ----------------------

    discordant_links = int(N) // 2

    a_file = False
    for f in os.listdir():
        fs = f.split('_')

        if fs[0] == 'corr':
            a_file = True 

            k = fs[1][1:]
            s = fs[2][1:]
            print(f'... processing seed {s} ...')

            df = pd.read_csv(f)

            sorted = df['Correct'].to_list()
            sorted.sort()
            pairs = len(sorted) # pairs of nodes (links)

            # construct the temporal series of "correctly identified links"
            # lists: x=time of the change, y=value of the "wrong links"
            x, y = [], []
            drop = 1
            for i in range(len(sorted) - 1): # corresponds to the number of nodes
                if sorted[i+1] == sorted[i]:
                    drop += 1
                    continue
                else:
                    x.append(sorted[i])
                    y.append(pairs - drop if len(y) == 0 else y[-1] - drop)
                    drop = 1
            # manage last sample
            if sorted[-1] == sorted[-2]:
                x.append(sorted[-2])
                y.append(pairs - drop if len(y) == 0 else y[-1] - (drop+1))
            else:
                x.append(sorted[-1])
                y.append(pairs - drop if len(y) == 0 else y[-1] - 1)

            # make the function piecewise constant to be able to average it
            # over the sample paths (seed)
            ts = []
            x_idx = 0
            for time in range(H):
                if time < x[0]:
                    ts.append(0)
                else:
                    if time == x[x_idx]: # new value of function
                        ts.append(y[x_idx])
                        x_idx += 1
                        if x_idx >= len(x):
                            # no anymore variation, exted the value till then end
                            for ttime in range(time+1, H):
                                ts.append(ts[-1])
                            break
                    else:
                        ts.append(ts[-1]) # make it piecewise const
            
            assert len(ts) == H

            # save the (preprocessed) ts in a dataframe (for all K)
            # Plot only later, reading from the dataframe
            row = {
                # 'Nodes': N,
                # 'r': r,
                'k': k,
                # 'sigma': sigma,
                # 'epsilon': epsilon,
                # 'delta': delta,
                'seed': s,
                # 'Exp': exp,
                'Time series': ts
            }
            row_df = pd.DataFrame([row])

            if os.path.exists(file_path):
                row_df.to_csv(file_path, mode='a', header=False, index=False, sep='\t') # append
            else:
                row_df.to_csv(file_path, mode='w', header=True, index=False, sep='\t')  # create

    if not a_file:
        raise RuntimeError('No files have been processed')

    # ------------------------- Actual plot from csv -------------------------

    fig, ax = plt.subplots(figsize=(9,5))

    df_sum = pd.read_csv(file_path, sep='\t')
    df_sum['Time series'] = df_sum['Time series'].map(ast.literal_eval) # from 'str' to 'list'

    grouped = df_sum.groupby('k')['Time series'].apply(lambda x: np.array(x.tolist())) # apply to each group
                                                                                       # Series to list of lists
                                                                                       # then numpy array
    # produce the curve for each explored dimension (K)
    for g_idx, (param, values) in enumerate(grouped.items()):

        if plot_path:
            for v in values:
                if norm_flag:
                    ax.plot(range(len(v)), v / discordant_links, label=f'K = {param}', color=color_list[g_idx])
                else: 
                    ax.plot(range(len(v)), v, label=f'K = {param}', color=color_list[g_idx])

        else:
            # find average over seeds and confidence interval width
            if norm_flag:
                mean, margin_of_error = calculate_confidence_interval(values / discordant_links)
            else:
                mean, margin_of_error = calculate_confidence_interval(values)
            
            time = range(len(mean))

            ax.plot(time, mean, label=f'K = {param}')
            ax.fill_between(time, mean - margin_of_error, mean + margin_of_error, alpha=0.6)


    # ax.set_ylabel(r'$\text{# Wrong Connections}$')
    ax.set_ylabel(r'$\text{Wrong Links}$')
    ax.set_xlabel(r'$t$')
    ax.legend()
    if log_flag:
        ax.set_yscale('log')
        if norm_flag:
            # ax.set_ylim([1e-5, 1])
            ax.set_ylim([1e-4, 1.1])
    plt.tight_layout()
    plt.show()
