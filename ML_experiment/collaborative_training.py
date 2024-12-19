# Program that implements the FL-DG Algorithm (Appendix H of the supplemental
# material and last experiment of Section 6). 
# Each node receives a new sample and constructs two overlapping minibatches
# over which trains for E epochs. The collaborative graph is a complete graph
# nodes are assigned to either one of two classes based on their node idx.
# Save info with 'wandb' and some info related to the structure as .csv file.

import torch
import torch.nn as nn
from torch import nn, optim
from torch.nn.functional import cosine_similarity
from torchvision import datasets, transforms
from torch.utils.data import Subset
import wandb
import argparse
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


# subclass MNIST dataset to create a new dataset with swapped labels "1" and "7"
# and "3" with "5"
class SwappedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if target == 3:
            target = 5
        elif target == 5:
            target = 3
        if target == 1:
            target = 7
        elif target == 7:
            target = 1
        return img, target


# this is a (very) simple neural network structure which will be initially used
# to test the collaborative training algorithm. It has only three layers
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        # using the crossentropyloss combines already a logSoftmax and a N
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def parse_args():
    parser = argparse.ArgumentParser(description='Collaborative Training')
    parser.add_argument(
        '-n', '--num_nodes',
        type=int,
        default=40,
        help='Number of computational units'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=123,
        help='Random seed of the random generator'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=1,
        help='Number of epochs for the training'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=int,
        default=0.01,
        help='Learning rate for the gradient descent'
    )
    parser.add_argument(
        '-th', '--threshold',
        type=float,
        default=0.76,
        help='Threshold for the cosine similarity'
    )
    parser.add_argument(
        '-eps', '--epsilon',
        type=float,
        default=0.008,
        help='Threshold for the norm of the difference of the weights'
    )
    parser.add_argument(
        '-v', '--verbose',
        default=False,
        action='store_true',
        help='Prints the mini-batch accuracy and loss'
    )

    args = parser.parse_args()

    if args.num_nodes % 2 != 0:
        raise ValueError('ERROR: The number of nodes must be even')

    return args


def train_model(model, criterion, optimizer, data_val, target_val, verbose=False):

    total_correct, total_samples = 0, 0

    model.train() # set the model to training mode
    optimizer.zero_grad()
    output = model(data_val)
    loss = criterion(output, target_val)
    loss.backward()
    optimizer.step()

    # compute TRAIN accuracy
    _, predicted = torch.max(output.data, 1)
    total_correct += (predicted == target_val).sum().item()
    total_samples += target_val.size(0)

    if verbose:
        print(f'Training accuracy: {100.0 * total_correct / total_samples}%')

    return 100.0 * total_correct / total_samples, loss.item()
    

# returns a list with non-overlapping splits of the 'dataset'
def non_overlapping_split(dataset, num_nodes):
    indices = torch.randperm(len(dataset)) # shuffle the dataset
    dataset = torch.utils.data.Subset(dataset, indices)

    # split the dataset into num_nodes parts
    return torch.utils.data.random_split(dataset, 
                                         [len(dataset) // num_nodes] * num_nodes)


# in a dictionary whose values are lists, homogenize the length of the lists
def pad_col_with_nan(d_p): # dict by reference

    max_len = max(len(lst) for lst in d_p.values()) # max lists length

    for key in d_p:
        len_diff = max_len - len(d_p[key])
        if len_diff > 0:
            d_p[key] += [np.nan] * len_diff # pad with nan



# ************************************ MAIN ************************************

if __name__ == "__main__":

    args = parse_args()
    do_not_print_norms = True

    torch.manual_seed(args.seed) # set random number generator's seed

    # ******************************** DATASET *********************************

    # torch.FloatTensor is in  "channel-first" format - Convert it and normalize
    # the greyscale values to [-1, +1]. 60000 training and 10000 test points.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load the (original) MNIST dataset
    D1_train = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=True,
        transform=transform
    )
    D1_test = datasets.MNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=False,
        transform=transform
    )

    # load the (swapped) MNIST dataset
    D2_train = SwappedMNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=True,
        transform=transform
    )
    D2_test = SwappedMNIST(
        '~/.pytorch/MNIST_data/',
        download=True,
        train=False,
        transform=transform
    )


    # ******************************** NETWORK *********************************

    complete_groups = True # if True consider a complete graph
    r = 5                  # degree of the random regular graph

    one_every = 20
    indices = list(range(0, len(D1_train), one_every)) # subsample the dataset
    D1_train_subset = Subset(D1_train, indices)

    indices = list(range(0, len(D2_train), one_every))
    D2_train_subset = Subset(D2_train, indices)

    n_comp = args.num_nodes // 2
    max_nonover_minibatch = 2
    batch_size = len(D1_train_subset) // (n_comp * max_nonover_minibatch)
    print(f'\tBatch Size: {batch_size} imposed by N={args.num_nodes} and {one_every}')

    G = nx.complete_graph(args.num_nodes)

    edge_to_idx = {e: idx for idx, e in enumerate(G.edges())} # for efficient saving

    c1_train_data = non_overlapping_split(D1_train_subset, n_comp)
    c2_train_data = non_overlapping_split(D2_train_subset, n_comp)

    # to have node 0 not on the border between communities
    edges_between_communities = []
    for i in range(n_comp):
        edges_between_communities.extend([(i,n_comp+j) for j in range(n_comp)])

    benchmark_edges = list(G.edges())[:len(edges_between_communities)]

    if args.verbose:
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, ax=ax)
        plt.show()


    # ****************************** INIT MODELS *******************************

    # each node will have the same-structure three-layer neural network model

    # initialize the first model and force all other models to start with the 
    # same weights (so that they all start from the same operational point)
    init_model = SimpleNN(784, 100, 10)

    for i in range(0, args.num_nodes): # copy  weights, start from the same point

        # crossentropyloss combines already a logSoftmax and a NLLLoss
        criterion = nn.CrossEntropyLoss() # specify loss function
        G.nodes[i]['criterion'] = criterion

        G.nodes[i]['tmp_list'] = {} # keep all the models for all the neighbors
        G.nodes[i]['opt_list'] = {}

        for idx, j in enumerate(G.neighbors(i)):
            G.nodes[i]['tmp_list'][j] = SimpleNN(784, 100, 10)
            G.nodes[i]['tmp_list'][j].load_state_dict(init_model.state_dict())

            G.nodes[i]['opt_list'][j] = \
                optim.SGD(G.nodes[i]['tmp_list'][j].parameters(), lr=args.learning_rate)

        G.nodes[i]['local_model'] = SimpleNN(784, 100, 10)
        G.nodes[i]['local_model'].load_state_dict(init_model.state_dict())

        G.nodes[i]['local_optimizer'] = optim.SGD(
            G.nodes[i]['local_model'].parameters(), lr=args.learning_rate
        )

        G.nodes[i]['avg_model'] = SimpleNN(784, 100, 10)
        G.nodes[i]['avg_model'].load_state_dict(init_model.state_dict())

        G.nodes[i]['avg_optimizer'] = optim.SGD(
            G.nodes[i]['avg_model'].parameters(), lr=args.learning_rate
        )

        G.nodes[i]['avg_model_no_cut'] = SimpleNN(784, 100, 10)
        G.nodes[i]['avg_model_no_cut'].load_state_dict(init_model.state_dict())

        G.nodes[i]['avg_optimizer_no_cut'] = optim.SGD(
            G.nodes[i]['avg_model_no_cut'].parameters(), lr=args.learning_rate
        )

        G.nodes[i]['accuracy'], G.nodes[i]['loss'] = [], []
        G.nodes[i]['local_accuracy'], G.nodes[i]['local_loss'] = [], []
        G.nodes[i]['accuracy_no_cut'], G.nodes[i]['loss_no_cut'] = [], []

        if (i - n_comp < 0):
            G.nodes[i]['train_data'] = c1_train_data[i]
        else:
            G.nodes[i]['train_data'] = c2_train_data[i - n_comp]


    # initialize "link models" described by the values of the parameters (then
    # they are trained by each node over their particular datasets)
    for u, v in G.edges():

        G[u][v]['link_model_weights'] = init_model.state_dict()
        G[u][v]['avg_similarity'] = np.nan


    # ************************ TRAIN AND COLLABORATE ***************************

    new_samples_per_iter = 1 # new samples per iteration
    add_round_comm = 0       # how many 'non-onlin' rounds to do

    cos_sim_evolution = {e: [] for e in G.edges()}
    edge_remove_time = {}
    edges_to_exclude = [] # (unordered) links not to consider in 'avg_model'

    n_mini_batches = len(G.nodes[0]['train_data']) // batch_size

    # for each node split the assigned dataset in two parts
    for node in range(args.num_nodes):
        init_size = batch_size
        remaining_size = len(G.nodes[node]['train_data']) - init_size
        G.nodes[node]['init_dataset'], G.nodes[node]['remaining'] = torch.utils.data.random_split(
            G.nodes[node]['train_data'], [init_size, remaining_size]
        )

    # print((len(G.nodes[0]['train_data']) - batch_size) // new_samples_per_iter)
    # print(len(G.nodes[0]['train_data']))
    # print(batch_size)

    dict_groups_types = {
        True: 'complete', 
        False: 'random'
    }

    run_id = 'run-1'
    run = wandb.init(
        project='collaborative-training-all-connected', 
        name=run_id, 
        config={
            'groups type': dict_groups_types[complete_groups],
            'number of nodes': args.num_nodes,
            '# swapped digits': 2,
            'dataset subsample - one every': one_every,
            'learning rate': args.learning_rate, 
            'batch size': batch_size,
            'new samples per iteration': new_samples_per_iter,
            'additional round of communication': add_round_comm,
            'epochs': args.epochs,
            'threshold': args.threshold,
            'epsilon': args.epsilon,
            'seed': args.seed,
        }
    )

    max_new_samples = ((len(G.nodes[0]['train_data']) - batch_size) // new_samples_per_iter)
    for idx_new in range(1, max_new_samples + 1 + add_round_comm):

        idx_new_sample = idx_new
        if idx_new_sample >= max_new_samples + 1: # no more samples
            idx_new_sample = max_new_samples

        print("*{:^48}*".format(" TRAIN AND COLLABORATE - ADD SAMPLE " + str(idx_new)))

        # I want to plot how many links have been removed (in total) and how
        # many of the wrong links have been removed
        print("\tTot links removed {}, {}/{} of the wrong: ".format(
                len(edges_to_exclude),
                sum([1 for e_set in edges_to_exclude if e_set in [set(el) for el in edges_between_communities]]),
                len(edges_between_communities)
                ))

        # means over batches, epochs and nodes
        training_acc = {'local': [], 'avg': [], 'avg_no_cut': []}
        loss_vals = {'local': [], 'avg': [], 'avg_no_cut': []}

        for node in tqdm(range(args.num_nodes)):

            # ********************** TRAIN NODE'S MODEL ************************
            for epoch in range(args.epochs):

                # add the new sample (for all nodes)
                updated_dataset = torch.utils.data.ConcatDataset([
                    G.nodes[node]['init_dataset'],
                    torch.utils.data.Subset(G.nodes[node]['remaining'],
                                            range(0, idx_new_sample*new_samples_per_iter))
                ])
                    
                # divide "statically" the dataset in two parts, check if the
                # second part becomes bigger than the first (whose size is fixed)
                two_mini = [
                    torch.utils.data.Subset(updated_dataset, range(0, batch_size)),
                    torch.utils.data.Subset(updated_dataset, range(len(updated_dataset) - batch_size, len(updated_dataset)))
                ]
                if len(two_mini[1]) > len(two_mini[0]):
                    print('WRNING: the second mini-batch is bigger than the first one!')

                for m in two_mini:

                    # create the dataloader
                    loader = torch.utils.data.DataLoader(m, batch_size=len(m))
            
                    # PHASE 1: each node trains a model on the available mini-batch data

                    # load the datasets
                    data, target = next(iter(loader))   # get the first (and only) batch
                    data = data.view(data.shape[0], -1) # flatten data (as it is 1 (color channel) * 28*28 (the image))

                    # here I always need to train only the local model (BENCHMARK)
                    try:
                        local_train_acc, local_loss = train_model(
                            G.nodes[node]['local_model'], 
                            G.nodes[node]['criterion'], 
                            G.nodes[node]['local_optimizer'], 
                            data, target, args.verbose
                        )
                        training_acc['local'].append(local_train_acc)
                        loss_vals['local'].append(local_loss)
                    except StopIteration: # skip if node has fewer mini-bathces
                        continue

                    # train the average model (FL-DG) to be updated with averages at end of cycle
                    try:
                        model_train_acc, model_loss = train_model(
                            G.nodes[node]['avg_model'], 
                            G.nodes[node]['criterion'], 
                            G.nodes[node]['avg_optimizer'], 
                            data, target, args.verbose
                        )
                        training_acc['avg'].append(model_train_acc)
                        loss_vals['avg'].append(model_loss)
                    except StopIteration: # skip if node has fewer mini-bathces
                        continue

                    try:
                        nocut_train_acc, nocut_loss = train_model(
                            G.nodes[node]['avg_model_no_cut'], 
                            G.nodes[node]['criterion'], 
                            G.nodes[node]['avg_optimizer_no_cut'], 
                            data, target, args.verbose
                        )       
                        training_acc['avg_no_cut'].append(nocut_train_acc)
                        loss_vals['avg_no_cut'].append(nocut_loss)
                    except StopIteration: # skip if node has fewer mini-bathces
                        continue


                    # train the 'link' model with the current piece of dataset
                    for neigh in G.neighbors(node): # use the ID of the link as key in the dict

                        # download the link model (updated with the mean average of the two endpoints)
                        G.nodes[node]['tmp_list'][neigh].load_state_dict(G[node][neigh]['link_model_weights'])

                        try:
                            train_model(
                                G.nodes[node]['tmp_list'][neigh], 
                                G.nodes[node]['criterion'], 
                                G.nodes[node]['opt_list'][neigh], 
                                data, target, args.verbose
                            )                    
                        except StopIteration: # skip if node has fewer mini-bathces
                            continue

        # at the end of each epoch training collect the aggregated (over minibatch, over epoch and over nodes)
        # average train accuracy for the three model
        wandb.log({f'ML-C-COLME TRAIN accuracy': sum(training_acc['avg']) / len(training_acc['avg'])})
        wandb.log({f'ML-C-COLME TRAIN loss': sum(loss_vals['avg']) / len(loss_vals['avg'])})
        wandb.log({f'Local TRAIN accuracy': sum(training_acc['local']) / len(training_acc['local'])})
        wandb.log({f'Local TRAIN loss': sum(loss_vals['local']) / len(loss_vals['local'])})
        wandb.log({f'Model-NO-CUT TRAIN accuracy': sum(training_acc['avg_no_cut']) / len(training_acc['avg_no_cut'])})
        wandb.log({f'Model-NO-CUT TRAIN loss': sum(loss_vals['avg_no_cut']) / len(loss_vals['avg_no_cut'])})

        # ****************************** DISCOVERY *********************************
        # after looping over all available minibatches and after E epochs (comm round)
                
        #  PHASE 2: each node updates its model with the neighbors' information
                                            
        for u, v in G.edges(): # check cosine similarity on links

            if {u,v} not in edges_to_exclude:

                link_weights = G[u][v]['link_model_weights'] # symmetric

                delta_u = torch.cat([
                    (param - link_weights[name]).flatten()
                    for name, param in G.nodes[u]['tmp_list'][v].named_parameters()
                ])

                delta_v = torch.cat([
                    (param - link_weights[name]).flatten()
                    for name, param in G.nodes[v]['tmp_list'][u].named_parameters()
                ])

                norm1 = torch.norm(delta_u) # big-small
                norm2 = torch.norm(delta_v) # small-big

                if norm1 > args.epsilon and norm2 > args.epsilon: # We considered epsilon < 0!
                
                    cosine_sim = cosine_similarity(
                        delta_u, delta_v, dim=0
                    ).item()

                    if np.isnan(G[u][v]['avg_similarity']): # mage initialization
                        G[u][v]['avg_similarity'] = cosine_sim

                    else:
                        samples = idx_new
                        G[u][v]['avg_similarity'] = (samples/(samples+1))*G[u][v]['avg_similarity'] + \
                                                        (1/(samples+1))*cosine_sim
                        
                        if (u,v) in benchmark_edges or (v,u) in benchmark_edges or \
                            (u,v) in edges_between_communities or (v,u) in edges_between_communities:

                            # links are undirected, by convention let's do: u < v
                            new_u = min(u,v)
                            new_v = max(u,v)
                            wandb.log({f'link {(new_u,new_v)}': G[new_u][new_v]['avg_similarity']})
                            if not do_not_print_norms:
                                wandb.log({
                                    f'norm 1 {(new_u,new_v)}': norm1,
                                    f'norm 2 {(new_u,new_v)}': norm2,
                                })
                        
                    # Update cosine similarity evolution
                    cos_sim_evolution[(u, v)].append(G[u][v]['avg_similarity'])

                    if G[u][v]['avg_similarity'] < args.threshold:
                        print(f'\tnode {u} ({norm1}) EXCLUDED node {v} ({norm2}) with cosine similarity {cosine_sim}\n')

                        if {u,v} not in edges_to_exclude:
                            edges_to_exclude.append({u,v})
                        edge_remove_time[(u,v)] = idx_new

                else:
                    cos_sim_evolution[(u, v)].append(G[u][v]['avg_similarity'])


        # ************************** COLLABORATION *****************************
        #            Agents use the new info to improve their models
        
        for node in tqdm(range(args.num_nodes)):

            opt_neighborhood = G.degree(node) + 1 # always consider all links 
            weights_sum = {
                name: param.clone()
                for name, param in G.nodes[node]['avg_model_no_cut'].named_parameters()
            } # init the collective param, considering params of current node

            for neigh in G.neighbors(node):
                for name, param in G.nodes[neigh]['avg_model_no_cut'].named_parameters():
                    weights_sum[name] += param

            avg_weights = {
                name: param / (opt_neighborhood)
                for name, param in weights_sum.items()
            }

            # use the average weights to update the model for node
            for name, param in G.nodes[node]['avg_model_no_cut'].named_parameters():
                param.data = avg_weights[name] # safe with tensors


            # ************** Model that excludes the 'wrong' links *************
                
            opt_neighborhood = 1 # the 'self-model'
            weights_sum = {
                name: param.clone()
                for name, param in G.nodes[node]['avg_model'].named_parameters()
            } # init the collective param, considering params of current node

            for neigh in G.neighbors(node):
                if {node, neigh} not in edges_to_exclude:
                    for name, param in G.nodes[neigh]['avg_model'].named_parameters():
                        weights_sum[name] += param
                    opt_neighborhood += 1

            avg_weights = {
                name: param / (opt_neighborhood)
                for name, param in weights_sum.items()
            }

            # use the average weights to update the model for node
            for name, param in G.nodes[node]['avg_model'].named_parameters():
                param.data = avg_weights[name] # safe with tensors


        # *********************** UPDATE LINK MODELS ***************************
                
        for u, v in G.edges(): # needed only to exclude the "wrong" links
            
            if {u,v} not in edges_to_exclude:

                # Update the link model with the average of the locally trained models
                weights_1 = G.nodes[u]['tmp_list'][v].state_dict()
                weights_2 = G.nodes[v]['tmp_list'][u].state_dict()

                avg_state_dict = {
                    name: (param1 + param2) / 2.0 
                    for (name, param1), (_, param2) in zip(weights_1.items(), weights_2.items())
                }

                G[u][v]['link_model_weights'] = copy.deepcopy(avg_state_dict)


        #******************** EVALUATE MODEL'S PERFORMANCES ********************
        #                      at each communication step
        
        # for each node compare the performance of local and collaborative model
        # perform the evaluation over batches and then average the metrics
                
        if (idx_new-1) % 1 == 0: # test the models at every instant

            for i in tqdm(range(args.num_nodes)): # test each node's model
                
                minibatch_acc, minibatch_loss = [], []
                minibatch_local_acc, minibatch_local_loss = [], []
                minibatch_nocut_loss, minibatch_nocut_acc = [], []

                if i // n_comp == 0:
                    test_iterator = iter(torch.utils.data.DataLoader(
                        D1_test,
                        batch_size=batch_size,
                        shuffle=True)
                    )
                else:   
                    test_iterator = iter(torch.utils.data.DataLoader(
                        D2_test,
                        batch_size=batch_size,
                        shuffle=True)
                    )

                G.nodes[i]['avg_model'].eval()        # eval mode
                G.nodes[i]['avg_model_no_cut'].eval() # eval mode
                G.nodes[i]['local_model'].eval()      # eval mode

                while True:
                    with torch.no_grad():  # disable gradient computation

                        try:
                            test_data, test_target = next(test_iterator)
                        except StopIteration:
                            break # exit when finish the miibatches

                        test_data = test_data.view(test_data.shape[0], -1) # flatten data

                        # evaluate loss and accuracy of the averae model
                        output = G.nodes[i]['avg_model'](test_data)
                        loss = G.nodes[i]['criterion'](output, test_target)
                        _, predicted = torch.max(output.data, 1) 
                        correct = predicted.eq(test_target.data.view_as(predicted)).sum()
                        accuracy = 100. * correct / len(test_data)
                        minibatch_loss.append(loss.item())
                        minibatch_acc.append(accuracy.item())

                        # do the same for the 'local' model
                        output_local = G.nodes[i]['local_model'](test_data)
                        local_loss = G.nodes[i]['criterion'](output_local, test_target)
                        _, predicted_local = torch.max(output_local.data, 1) 
                        correct_local = predicted_local.eq(test_target.data.view_as(predicted_local)).sum()
                        local_accuracy = 100. * correct_local / len(test_data)
                        minibatch_local_loss.append(local_loss.item())
                        minibatch_local_acc.append(local_accuracy.item())

                        # DO this for the no-cut model
                        output_nocut = G.nodes[i]['avg_model_no_cut'](test_data)
                        nocut_loss = G.nodes[i]['criterion'](output_nocut, test_target)
                        _, predicted_nocut = torch.max(output_nocut.data, 1) 
                        correct_nocut = predicted_nocut.eq(test_target.data.view_as(predicted_nocut)).sum()
                        nocut_accuracy = 100. * correct_nocut / len(test_data)
                        minibatch_nocut_loss.append(nocut_loss.item())
                        minibatch_nocut_acc.append(nocut_accuracy.item())
                        

                G.nodes[i]['loss'].append(np.mean(minibatch_loss))
                G.nodes[i]['accuracy'].append(np.mean(minibatch_acc))

                G.nodes[i]['local_loss'].append(np.mean(minibatch_local_loss))
                G.nodes[i]['local_accuracy'].append(np.mean(minibatch_local_acc))

                G.nodes[i]['loss_no_cut'].append(np.mean(minibatch_nocut_loss))
                G.nodes[i]['accuracy_no_cut'].append(np.mean(minibatch_nocut_acc))

            avg_collaborative_accuracy = sum(node['accuracy'][-1] for node in G.nodes.values()) / len(G.nodes)
            avg_local_accuracy = sum(node['local_accuracy'][-1] for node in G.nodes.values()) / len(G.nodes)
            avg_nocut_accuracy = sum(node['accuracy_no_cut'][-1] for node in G.nodes.values()) / len(G.nodes)
            print('avg accuracy {:.4f}, avg local {:.4f}, avg nocut {:.4f}'.format(avg_collaborative_accuracy, avg_local_accuracy, avg_nocut_accuracy))
            wandb.log({'ML-C-COLME Accuracy': avg_collaborative_accuracy})
            wandb.log({'LOCAL Accuracy': avg_local_accuracy})
            wandb.log({'Model-NO-CUT Accuracy': avg_nocut_accuracy})


    run.finish()


    # ************************* SAVE PERFORMANCE METRICS ***************************

    # save the values collected (to avoid recomputation)
    # I want to save the cosine similarities and the performances of the models

    pad_col_with_nan(cos_sim_evolution)
    cos_sim_evolution = {str(k): v for k, v in cos_sim_evolution.items()}
    df1 = pd.DataFrame(cos_sim_evolution)
    df1.to_csv(f'cos_sim_{run_id}.csv', index=False, sep='\t')

    dict_df = {}
    for i in range(args.num_nodes):
        dict_df['accuracy_node_' + str(i)] = G.nodes[i]['accuracy']
        dict_df['loss_node_' + str(i)] = G.nodes[i]['loss']
        dict_df['local_accuracy_node_' + str(i)] = G.nodes[i]['local_accuracy']
        dict_df['local_loss_node_' + str(i)] = G.nodes[i]['local_loss']
        dict_df['nocut_accuracy_node_' + str(i)] = G.nodes[i]['accuracy_no_cut']
        dict_df['nocut_loss_node_' + str(i)] = G.nodes[i]['loss_no_cut']
    
    pad_col_with_nan(dict_df)
    df2 = pd.DataFrame(dict_df)
    df2.to_csv(f'perf_no_cut_{run_id}.csv', index=False)


    edge_remove_time = {str(k): [v] for k, v in edge_remove_time.items()}
    df3 = pd.DataFrame(edge_remove_time)
    df3.to_csv(f'edge_rem_{run_id}.csv', index=False, sep='\t')
