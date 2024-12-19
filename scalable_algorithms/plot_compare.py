import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import scipy.stats as stats # ci

def parse_args():
    parser = argparse.ArgumentParser(description="Average performance evaluation")
    parser.add_argument('-l', '--log', action='store_true', help='log scale and error')
    parser.add_argument('-s', '--subfolder_list', metavar='s1 .. sN', nargs='+', help='Subfolder from which perform the comparison')
    parser.add_argument('--no_title', action='store_true', help='Does not show the title of the plot')
    parser.add_argument('--wrong_abs', action='store_true', help='Absolute value of wrong nodes')
    parser.add_argument('-r', '--n_neigh', type=int, default=10, help='Number of neighbors to be considered')
    
    return parser.parse_args()

sns.set_theme(context="talk", style="ticks", palette="deep", font="sans-serif",\
                color_codes=True, font_scale=0.8,
                rc={'figure.facecolor': 'white', "font.family": "sans-serif", 
                    'axes.labelpad': 8, 'legend.fontsize':13,
                    'lines.markersize': 8, 'lines.linewidth': 0.8,
                    'lines.linestyle': '--', 'lines.marker': 'o', 'lines.markersize': 2})

sns.set_style('whitegrid') # add a grid to the plots

sns.set_palette('tab10') # distinguishable colors

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


args = parse_args()


# default params to be used if more scenarios for each algorithm are present
alpha_compare = 0.5
up_to_compare = 5
int_avg_deg = args.n_neigh

df_list = []
for subfolder in args.subfolder_list:
    data_path = os.path.join(os.getcwd(), subfolder, 'preliminary.csv')

    if os.path.exists(data_path):
        df_tmp = pd.read_csv(data_path)

        # filter the datasets
        df_tmp = df_tmp[df_tmp['int_avg_deg'] == int_avg_deg]

        alg_in_df = df_tmp['alg'].unique()
        df_tmp_list = []
        for alg_v in alg_in_df:
            if alg_v == 'belief_propagation_v1':
                df_tmp_list.append(df_tmp[(df_tmp['alg'] == alg_v) & (df_tmp['up_to'] == up_to_compare)])

            elif alg_v == 'consensus':
                df_tmp_list.append(df_tmp[(df_tmp['alg'] == alg_v) & (df_tmp['alpha'] == alpha_compare)])

            elif alg_v == 'colme' or alg_v == 'colme_recompute':
                df_tmp_list.append(df_tmp[(df_tmp['alg'] == alg_v)])
        
        if df_tmp_list != []:
            df_tmp = pd.concat(df_tmp_list, ignore_index=True)

        df_list.append(df_tmp)
    else:
        raise Exception(f'FATAL ERROR: file {data_path} not found.')
    
df = pd.concat(df_list, ignore_index=True)


# COL: ['alg', 'seed', 'n', 'int_avg_deg', 'sigma', 'alpha', 'up_to', 'epsilon',
#       'delta', 'iter', 'pe', 'pl', 'po', 'pc', 'lost_neigh', 'wrong_neigh']

non_seed_col  = ['alg', 'n', 'int_avg_deg', 'sigma', 'alpha', 'up_to',
                'epsilon', 'delta', 'iter']

non_seed_iter = ['alg', 'n', 'int_avg_deg', 'sigma', 'alpha', 'up_to',
                'epsilon', 'delta']

to_avg_col    = ['pe', 'lost_neigh', 'wrong_neigh']

to_plot_col   = ['pe']

# average over the output and compute the entries (seed) over we are averaging
# do this for each discrete time instat (iter)
df_avg = df.groupby(non_seed_col)[to_avg_col].agg(
                ['mean', 'count', stats.sem]).reset_index() # this converts back to a df

df_avg.columns = ['{}_{}'.format(col1, col2) if col2 else col1 
                     for col1, col2 in df_avg.columns.values]

# calculate the confidence interval, using standard error (sem)
confidence_level = 0.95
for col in to_avg_col:
    df_avg[f'{col}_ci'] = stats.t.ppf((1 + confidence_level) / 2., df_avg[f'{col}_count'] - 1) * df_avg[f'{col}_sem']

# get a dataframe with the scenarios from the experiments
scenarios = df[non_seed_iter].drop_duplicates().reset_index(drop=True)


# 'int_avg_deg': 'r' (is okay only assuming we are experimenting with GNR)
tit_dict = {
    'alg': 'algorithm',
    'n': 'N',
    'int_avg_deg': 'r',
    'alpha': r'$\alpha$',
    'up_to': r'$\kappa$',
    'epsilon': r'$\epsilon$',
    'delta': r'$\delta$'
} # so far not used

out_dict = {
    'pe': 'Algorithm',
    'pl': 'Local Estimate',
    'po': 'Oracle',
    'pc': 'On Two-Largest-CC',
    'lost_neigh': 'missed neighbors',
    'wrong_neigh': 'wrong neighbors' 
}

alg_dict = {
    'belief_propagation_v1': 'B-COLME',
    'belief_propagation_v2': 'B-COLME',
    'consensus': 'C-COLME',
    'colme': 's-COLME',
    'colme_recompute': 'COLME'
}

color_dict = {
    'belief_propagation_v1': color_list[0],
    'consensus': color_list[1],
    'colme': color_list[2],
    'colme_recompute': color_list[3]
}


# two plots one over the other, top: error probability, bottom: wrong neighbors
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

for _, scenario in scenarios.iterrows():# loop over each scenario

    alg = scenario['alg'] # needed later for the colors

    # match the parameters of the scenario
    df_scenario = df_avg[
        (df_avg['alg'] == scenario['alg']) & 
        (df_avg['n'] == scenario['n']) & 
        (df_avg['int_avg_deg'] == scenario['int_avg_deg']) &
        (df_avg['sigma'] == scenario['sigma']) &
        (df_avg['alpha'] == scenario['alpha']) &
        (df_avg['up_to'] == scenario['up_to']) &
        (df_avg['epsilon'] == scenario['epsilon']) &
        (df_avg['delta'] == scenario['delta'])
    ]
    
    for col in to_plot_col:

        if args.log:
            upper_bound = 1.0 - (df_scenario[f'{col}_mean'] - df_scenario[f'{col}_ci'])
            lower_bound = 1.0 - (df_scenario[f'{col}_mean'] + df_scenario[f'{col}_ci'])
            ax1.plot(df_scenario['iter'], 1.0 - df_scenario[f'{col}_mean'], label=f'{alg_dict[alg]}', marker='', color=color_dict[alg])
            ax1.fill_between(df_scenario['iter'], lower_bound, upper_bound, color=color_dict[alg], alpha=.1)
        else:
            upper_bound = (df_scenario[f'{col}_mean'] - df_scenario[f'{col}_ci'])
            lower_bound = (df_scenario[f'{col}_mean'] + df_scenario[f'{col}_ci'])
            ax1.plot(df_scenario['iter'], df_scenario[f'{col}_mean'], label=f'{alg_dict[alg]}', marker='', color=color_dict[alg])
            ax1.fill_between(df_scenario['iter'], lower_bound, upper_bound, color=color_dict[alg], alpha=.1)

    if args.wrong_abs:
        upper_bound = (df_scenario['wrong_neigh_mean'] - df_scenario['wrong_neigh_ci'])
        lower_bound = (df_scenario['wrong_neigh_mean'] + df_scenario['wrong_neigh_ci'])
        ax2.plot(df_scenario['iter'], df_scenario['wrong_neigh_mean'], marker='', color=color_dict[alg])
        ax2.fill_between(df_scenario['iter'], lower_bound, upper_bound, color=color_dict[alg], alpha=.1)
    else:
        normalized_wrong = df_scenario['wrong_neigh_mean'] / df_scenario['wrong_neigh_mean'].max()
        normalized_ci_wrong = df_scenario['wrong_neigh_ci'] / df_scenario['wrong_neigh_mean'].max()
        ax2.plot(df_scenario['iter'], normalized_wrong, marker='', color=color_dict[alg])
        ax2.fill_between(df_scenario['iter'], (normalized_wrong - normalized_ci_wrong), (normalized_wrong + normalized_ci_wrong), color=color_dict[alg], alpha=.1)
    
    alg_add_param = ''
    if scenario['alg'] == 'belief_propagation_v1' or scenario['alg'] == 'belief_propagation_v2':
        alg_add_param = r'$\kappa$={}'.format(scenario['up_to'])
    elif scenario['alg'] == 'consensus':
        alg_add_param = r'$\alpha$={}'.format(scenario['alpha'])
    elif scenario['alg'] == 'colme' or scenario['alg'] == 'colme_recompute':
        alg_add_param = 'v={}'.format(scenario['int_avg_deg'])
    else:
        sys.exit('FATAL ERROR: unknown algorithm.')

    scenario_info = r'{} ({}) over Gnr (r={}) with N={} and $\sigma$={}, with $\epsilon$={}, $\delta$={}'.format(
        alg_dict[scenario['alg']],
        alg_add_param, 
        scenario['int_avg_deg'],
        scenario['n'],
        scenario['sigma'],
        scenario['epsilon'],
        scenario['delta']
    )

    if not args.no_title:
        fig.suptitle(scenario_info)

    if args.log:
        ax1.set_yscale('log')
        ax1.set_ylabel(r'$\mathrm{Error} \, \mathrm{Probability}$')
        ax1.set_ylim([1e-05, 1.5])

        if args.wrong_abs:
            ax2.set_ylim([-0.05, 1.1*(df_scenario['wrong_neigh_mean'].max())]) 
            ax2.set_ylabel(r'$\mathrm{Wrong} \, \mathrm{Neighbors}$')

        else:
            ax2.set_yscale('log')
            ax2.set_ylim([1e-05, 1.2])
            ax2.set_ylabel(r'$\frac{\mathrm{Wrong} \, \mathrm{Neigh}}{\max(\mathrm{Wrong} \, \mathrm{Neigh})}$')
    else:
        ax1.set_ylabel(r'$\mathrm{Success Probability}$')


    ax2.set_xlabel(r'$\mathrm{Iterations}$') # shared y label

    ax1.legend(loc='upper right')

plt.tight_layout()
plt.show()