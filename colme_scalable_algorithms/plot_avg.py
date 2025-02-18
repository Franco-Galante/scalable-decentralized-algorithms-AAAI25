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
    parser.add_argument('-s', '--subfolder', type=str, default='', help='Subfolder of the csv to plot')
    parser.add_argument('--no_title', action='store_true', help='Does not show the title of the plot')
    parser.add_argument('--wrong_abs', action='store_true', help='Absolute value of wrong nodes')
    
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

FONTSIZE = 16

data_path = os.path.join(os.getcwd(), args.subfolder, 'preliminary.csv')

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    raise Exception(f'FATAL ERROR: file {data_path} not found.')

n_seeds = df['seed'].nunique()
print(f'Average over {n_seeds} seeds')

# COL: ['alg', 'seed', 'n', 'int_avg_deg', 'sigma', 'alpha', 'up_to', 'epsilon',
#       'delta', 'iter', 'pe', 'pl', 'po', 'pc', 'lost_neigh', 'wrong_neigh']

non_seed_col  = ['alg', 'n', 'int_avg_deg', 'sigma', 'alpha', 'up_to',
                'epsilon', 'delta', 'iter']

non_seed_iter = ['alg', 'n', 'int_avg_deg', 'sigma', 'alpha', 'up_to',
                'epsilon', 'delta']

to_avg_col    = ['pe', 'pl', 'po', 'pc', 'lost_neigh', 'wrong_neigh']

to_plot_col   = ['pe', 'pl', 'po', 'pc']

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
    'pe': color_list[0],
    'pl': color_list[1],
    'po': color_list[2],
    'pc': color_list[3],
    'wrong_links': color_list[4]
}

for _, scenario in scenarios.iterrows():# loop over each scenario

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

    # two plots one over the other, top: error probability, bottom: wrong neighbors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    
    for col in to_plot_col:

        if col == 'pc' and (df_scenario['alg'].iloc[0] == 'colme' or df_scenario['alg'].iloc[0] == 'colme_recompute'):

            pass # do not plot the on 2-largest-cc for COLME

        else:
            if args.log:
                upper_bound = 1.0 - (df_scenario[f'{col}_mean'] - df_scenario[f'{col}_ci'])
                lower_bound = 1.0 - (df_scenario[f'{col}_mean'] + df_scenario[f'{col}_ci'])
                ax1.plot(df_scenario['iter'], 1.0 - df_scenario[f'{col}_mean'], label=f'{out_dict[col]}', marker='', color=color_dict[col])
                ax1.fill_between(df_scenario['iter'], lower_bound, upper_bound, color=color_dict[col], alpha=.1)
            else:
                upper_bound = (df_scenario[f'{col}_mean'] - df_scenario[f'{col}_ci'])
                lower_bound = (df_scenario[f'{col}_mean'] + df_scenario[f'{col}_ci'])
                ax1.plot(df_scenario['iter'], df_scenario[f'{col}_mean'], label=f'{out_dict[col]}', marker='', color=color_dict[col])
                ax1.fill_between(df_scenario['iter'], lower_bound, upper_bound, color=color_dict[col], alpha=.1)

    if args.wrong_abs:
        upper_bound = (df_scenario['wrong_neigh_mean'] - df_scenario['wrong_neigh_ci'])
        lower_bound = (df_scenario['wrong_neigh_mean'] + df_scenario['wrong_neigh_ci'])
        ax2.plot(df_scenario['iter'], df_scenario['wrong_neigh_mean'], marker='', color=color_dict['wrong_links'])
        ax2.fill_between(df_scenario['iter'], lower_bound, upper_bound, color=color_dict['wrong_links'], alpha=.1)
    else:
        normalized_wrong = df_scenario['wrong_neigh_mean'] / df_scenario['wrong_neigh_mean'].max()
        normalized_ci_wrong = df_scenario['wrong_neigh_ci'] / df_scenario['wrong_neigh_mean'].max()
        ax2.plot(df_scenario['iter'], normalized_wrong, marker='', color=color_dict['wrong_links'])
        ax2.fill_between(df_scenario['iter'], (normalized_wrong - normalized_ci_wrong), (normalized_wrong + normalized_ci_wrong), color=color_dict['wrong_links'], alpha=.1)
    
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
        ax1.set_ylabel(r'$\mathrm{Wrong} \, \mathrm{Estimates}$', fontsize=FONTSIZE)
        ax1.set_ylim([1e-05, 1.5])

        if args.wrong_abs:
            ax2.set_ylim([-0.05, 1.1*(df_scenario['wrong_neigh_mean'].max())]) 
            ax2.set_ylabel(r'$\mathrm{Wrong} \, \mathrm{Neighbors}$', fontsize=FONTSIZE)

        else:
            ax2.set_yscale('log')
            ax2.set_ylim([1e-05, 1.2])
            # ax2.set_ylabel(r'$\frac{\mathrm{Wrong} \, \mathrm{Neigh}}{\max(\mathrm{Wrong} \, \mathrm{Neigh})}$')
            ax2.set_ylabel(r'$\mathrm{Wrong} \, \mathrm{Links}$', fontsize=FONTSIZE)
    else:
        ax1.set_ylabel(r'$\mathrm{Success Probability}$', fontsize=FONTSIZE)


    ax2.set_xlabel(r'$\mathrm{t}$', fontsize=FONTSIZE) # shared y label

    ax1.legend(loc='upper right')
    # ax1.legend()
    
    # ax2.legend() à I do not have legend for the second plot
    plt.tight_layout()
    plt.show()

if not os.path.exists('figs'):
    os.makedirs('figs')
fig.savefig(os.path.join('figs', f'{args.subfolder}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join('figs', f'{args.subfolder}.png'), bbox_inches='tight')
