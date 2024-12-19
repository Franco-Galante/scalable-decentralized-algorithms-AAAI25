This experiment was done using Python 3.10.11 (+ requirements.txt) on Windows.

As a first step create the environment with a suitable python version and the 
python packages as for 'requirements.txt'. Then from command line:

python indep_pairs_discovery.py -n 10000 -r 10 -k K_VAL -t 7000 --seed SEED_VAL
(you may need to type 'python3' in Linux)

Repeat the experiment with the 'K_VAL' of choice (in the paper K=1,2,4,8) and
repeat the command (e.g., in a foreach loop) for 10  choices of SEED_VAL (in 
the paper 101, 111, 222, 333, 444, 555, 666, 777, 888, 999).
The script will launch (trough 'subprocess.run') the 'cc_size_est.py' script,
which estimates the size of the connected component in a random graph with 'N' 
nodes and 'r' average degree.

The results will be stored in a subfolder of the 'res' folder, whose name con-
tains the parameters of the experiment. Once all the results have be obtained,
navigate inside this folder and call the 'plot.py' script within this folder,
for example like:

python ..\..\plot.py
