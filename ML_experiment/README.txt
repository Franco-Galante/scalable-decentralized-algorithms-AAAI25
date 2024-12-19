To run the collaborative model Python (3.9.13) is needed.
To obtain the outputs (in csv files) run the following:

python .\collaborative_training.py --num_nodes 100 -t 0.1 -e 15 -eps -0.1

Then to plot the results (corresponding to Figure 3) first download from wandb
the accuracy csv (filter by 'accuracy' and download as csv the data for the test
accuracy, rename the files if necessary), then run:

python .\plot_online_ML.py