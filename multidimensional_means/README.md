# How to Perform the Experiments

This experiment uses Python 3.10.11 on Windows. To begin, ensure all necessary dependencies listed in `requirements.txt` are installed. Thus, create an appropriate virtual environment. Please use `.venv` as the name of the environment within the folder due to compatibility with the script which internally needs to call another Python script and searches for a `.venv` environment within the current folder.


### Obtain the Results

**Run the Command**

   From the command line, execute the following command, where `K_VAL` is the dimension of the vectorial space and `SEED_VAL` is a particular seed choice:
   ```bash
   python indep_pairs_discovery.py -n 10000 -r 10 -k $K_VAL -t 7000 --seed $SEED_VAL
   ```

To obtain the results in Appendix C of the paper (Figure 4), repeat the experiment with each the `K_VAL`s used in the paper (namely, $K=1,2,4,8$) and for 10 different choices of `SEED_VAL` (in particular, the seed values to use are: 101, 111, 222, 333, 444, 555, 666, 777, 888, 999).
Note that script (`indep_pairs_discovery.py`) will launch (trough `subprocess.run`) the `cc_size_est.py` script, which estimates the size of the connected component in a random graph with $N$ nodes and an average degree of $r$.

The results will be stored in a subfolder within the `res` folder (created at runtime). Each subfolder's name will include the parameters of the experiment, ensuring distinct subfolders for each experiment. 


### Plot the Results

Once all the results have be obtained, navigate inside the experiment folder and call the `plot.py` script from within this folder, for example:

   ```bash
   python ..\..\plot.py
   ```
