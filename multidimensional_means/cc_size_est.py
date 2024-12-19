import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse
import os


def parse_args():

    parser = argparse.ArgumentParser(
        description='Estimate the average size of the connected component of a node'
        )
    
    parser.add_argument(
        '--nodes', '-n',
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
        '--classes', '-c',
        type=int,
        default=2,
        help='Number of classes to discover'
        )
    
    parser.add_argument(
        '--erdos-renyi', '-er',
        action='store_true',
        help='Use the Erdos-Renyi random graph model'
        )
    
    parser.add_argument(
        '--num-trials',
        type=int,
        default=50,
        help='Number of trials to average the results'
        )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Prints outputs in addition to saving them'
        )
    
    return parser.parse_args()


def average_component_frac(n, r, er=False, num_trials=50):
    '''
    Compute the average size of the connected component a node uniformely picked
    up at random would see. (weigthed average with respct to the component size)
    '''
    avg_sizes = []
    for _ in tqdm(range(num_trials)):
        if er:
            p = r / (n-1)
            G = nx.fast_gnp_random_graph(n, p) # nx.erdos_renyi_graph(n, p)
        else:
            G = nx.gnp_random_graph(n, r)

        component_sizes = [len(c) for c in nx.connected_components(G)]
        avg_sizes.append((1/n**2) * sum([size**2 for size in component_sizes]))

    return np.mean(avg_sizes)


if __name__ == '__main__':

    args = parse_args()

    # NOTE: we make all considerations on average (divide r by # classes)
    avg_size = average_component_frac(args.nodes, args.r / args.classes, 
                                      args.erdos_renyi, args.num_trials)

    if args.verbose:
        print(f"Avg connected component for N={args.nodes} (r={args.r}): {avg_size:.2f}")

    if os.path.isfile('cc_size_est.csv'):
        with open('cc_size_est.csv', 'a') as f:
            f.write(f"{args.erdos_renyi},{args.classes},{args.nodes},{args.r},{avg_size}\n")

    else:
        with open('cc_size_est.csv', 'w') as f:
            f.write("ER,Classes,Nodes,r,AvgSize\n")
            f.write(f"{args.erdos_renyi},{args.classes},{args.nodes},{args.r},{avg_size}\n")
