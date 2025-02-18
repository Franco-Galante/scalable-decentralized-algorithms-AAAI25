# How to Perform the Experiments

This experiment uses Python 3.9.13 on Windows. To begin, ensure all necessary dependencies listed in `requirements.txt` are installed. Thus, create an appropriate virtual environment.

### Obtain the Results

**Run the Command**

   From the command line, execute the following command:
   ```bash
   python .\collaborative_training.py --num_nodes 100 -t 0.1 -e 15 -eps -0.1
   ```

### Plot the Results

Once the above script terminates, it is needed to download from Wandb the *accuracy* data in csv format. Go on `collaborative-training-all-connected` project on Wandb, which has been just produced, and filter by 'accuracy'. Then download `LOCAL Accuracy` and save it in this folder as `accuracy_local-1.csv`, `Model-NO-CUT Accuracy` as `accuracy_nocut-1.csv`, and lastly `ML/C/COLME Accuracy` as `accuracy_model-1.csv`. Now, the plotting script can be invoked, run:

```bash
python plot_online_ML.py
```
