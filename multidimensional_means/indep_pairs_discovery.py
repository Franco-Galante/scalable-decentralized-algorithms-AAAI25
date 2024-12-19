# NOTE: if you run it without the 'quick_run' flag set it will do all the 't' iterations
#       for the second half of the connections (concordant links), either do a 'quick_run'
#       or appropriately choose the time horizon. 

import os
import math
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time


def parse_args():

    parser = argparse.ArgumentParser(
        description='Time (n) to discover similarity classes [HP. indep pairs]'
        )
    
    parser.add_argument(
        '--nodes', '-n',
        metavar='N',
        type=int,
        default=10000,
        help='Number of nodes in the graph'
        )
    
    parser.add_argument(
        '-r',
        type=int,
        default=10,
        help='Number of neighbors of each node [Gnr random graph model]'
        )
    
    parser.add_argument(
        '--dimension', '-k',
        metavar='D',
        type=int,
        default=1,
        help='Dimension of the vectorial random variables to estimate'
        )
    
    parser.add_argument(
        '--time-horizon', '-t',
        metavar='H',
        type=int,
        default=2000,
        help='Considered iterations for the (discovery) algorithm'
        )
    
    parser.add_argument(
        '--epsilon', '-e',
        metavar='E',
        type=float,
        default=0.1,
        help='Error threshold for the algorithm'
        )
    
    parser.add_argument(
        '--delta',
        metavar='D',
        type=float,
        default=0.1,
        help='Confidence level for the algorithm'
        )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for the random number generator'
        )
    
    parser.add_argument(
        '--sigma', '-s',
        type=float,
        default=2.0,
        help='Standard deviation of the Gaussian distribution'
        )
    
    parser.add_argument(
        '--colme', '-c',
        default=False,
        action='store_true',
        help='Use pairs and gamma as for the original ColME algorithm'
        )

    return parser.parse_args()


def beta_int(n_p, sigma_p, gamma_p):

    ln_v = math.log(math.sqrt(n_p + 1) / gamma_p)
    sqrt_v = math.sqrt((2/n_p) * (1+1/n_p) * ln_v)

    return sigma_p * sqrt_v



if __name__ == '__main__':

    start = time.time()
    
    args = parse_args()

    run_sub = True    # to run the subprocess to populate 'cc_size_est.csv'
    same_start = True # deterministically divide in class 50%-50%
    quick_run = False # skips the "concordant links" (very unlikely to make mistakes)
    EXPERIMENT = 0    # 0: mu=1/sqrt(D), 1: mu=e1, 2: mu=rnd phase

    np.random.seed(args.seed) # set the seed of the simulator

    D = args.dimension
    H = args.time_horizon

    # means of the TWO classes
    class_to_mu = {
        0: 0.0, 
        1: 1 / math.sqrt(D) # guarantees distance = 1 (norm 2) [worst case]
        }                   

    E = math.ceil((args.nodes * args.r) / 2)
    if os.path.isfile('cc_size_est.csv'):
        df = pd.read_csv('cc_size_est.csv')
        avg_size = df.loc[(df['Nodes'] == args.nodes) &
                          (df['Classes'] == len(class_to_mu)) &
                          (df['r'] == args.r) & 
                          (df['ER'] == True), 'AvgSize'] # TODO: manage 'True' ER
        if not avg_size.empty:
            CC_a = avg_size.values[0] * args.nodes

        else:
            cmd_sub = [
                'python', 'cc_size_est.py',
                '--nodes', str(args.nodes),
                '-r', str(args.r),
                '--erdos-renyi',
                '--num-trials', '50'
                ]

            if run_sub:
                result = subprocess.run(cmd_sub, capture_output=True, text=True)

                if result.returncode != 0:
                    print("Error:", result.stderr)
                    raise RuntimeError('cc_size_est.py failed')
                else:
                    df = pd.read_csv('cc_size_est.csv')
                    avg_size = df.loc[(df['Nodes'] == args.nodes) &
                                    (df['Classes'] == len(class_to_mu)) &
                                    (df['r'] == args.r) & 
                                    (df['ER'] == True), 'AvgSize']
                    CC_a = avg_size.values[0]

            else:
                cmd_str = ' '.join(cmd_sub)

            raise RuntimeError(f'Entry not found in cc_size_est.csv. Run:\n {cmd_str}')
    else:
        raise FileNotFoundError('cc_size_est.csv not found')

    gamma = args.delta / (4.0 * args.r * CC_a)
    
    if args.colme:
        E = math.ceil(args.nodes * (args.nodes - 1) / 2)
        gamma = args.delta / (4.0 * args.nodes)

    g = gamma / D # adjusted for the dimension

    corr_decision, incorr_decision = [], []

    # NOTE: instead of randomly extracting the classes (N_class_1 = Bin(N, 0.5)) let's
    #       force the discordant links to be exactly N/2 (so all graphs starts equally)
    discordant_links = args.nodes // 2

    for pair in tqdm(range(E)): # we can do each pair independently
                                # NOTE: more correct with a Gnp model
        if same_start:
            if pair < discordant_links: # make the first half discordant
                a_class = 0
                b_class = 1
            elif not quick_run:
                a_class = np.random.binomial(1, 0.5) # make the class random
                b_class = a_class                    # make the link concordant

        else:
            a_class = np.random.binomial(1, 0.5) # class of the first node
            b_class = np.random.binomial(1, 0.5)

        est_a = 0.5 * np.random.normal(class_to_mu[a_class], args.sigma, D) + \
                0.5 * np.random.normal(class_to_mu[a_class], args.sigma, D)
        est_b = np.random.normal(class_to_mu[b_class], args.sigma, D)

        d_opt = np.abs(est_a - est_b) - beta_int(2, args.sigma, g) - beta_int(1, args.sigma, g)
        positive_component = np.any(d_opt > 0) # if any positive -> remove the link

        for n in range(2, H):

            # consider UNDIRECTED links, HP. symmetric pruning of links and look
            # at only one direction a -> b            

            est_a = (n / (n+1)) * est_a + (1 / (n+1)) * np.random.normal(class_to_mu[a_class], args.sigma, D)
            est_b = (n / (n+1)) * est_b + (1 / (n+1)) * np.random.normal(class_to_mu[b_class], args.sigma, D)

            d_opt = np.abs(est_a - est_b) - beta_int(n+1, args.sigma, g) - beta_int(n, args.sigma, g)
            positive_component = np.any(d_opt > 0)

            if positive_component and a_class!=b_class:
                # correct decision [irreversible]
                corr_decision.append(n)
                break

            elif positive_component and a_class==b_class:
                # incorrect decision [irreversible]
                incorr_decision.append(n)
                break

        # if arrived here -> no removal (d_opt < 0 for all iterations)
        if not positive_component and a_class==b_class:
            # nothing needed to be done (correct decision)
            corr_decision.append(0)
            
        elif not positive_component and a_class!=b_class:
            # may not yet have taken the decision -> never properly removed link
            corr_decision.append(H)

    print(corr_decision, '\n', incorr_decision)
    
    # manage output folder and subfolders
    if not os.path.isdir('res'): # create res folder if not already present
        os.makedirs('res')
    sub_res_folder = f'GN_{args.nodes}_r{args.r}_Exp{EXPERIMENT}_sigma{args.sigma}_d{args.delta}_e{args.epsilon}'
    if not os.path.isdir(os.path.join('res', sub_res_folder)):
        os.makedirs(os.path.join('res', sub_res_folder))

    file_suffix = f'k{D}_s{args.seed}'
    pd.DataFrame({'Correct': corr_decision}).to_csv(
        os.path.join('res', sub_res_folder, f'corr_{file_suffix}.csv'), index=False
        )   
    pd.DataFrame({'Incorrect': incorr_decision}).to_csv(
        os.path.join('res', sub_res_folder, f'incorr_{file_suffix}.csv'), index=False
        )

    print(f"Simulation time: {time.time() - start:.2f} s")

