## How to Perform the Experiments

To build and execute the project, ensure you have CMake installed. The code is compatible with both Windows and Linux (the experiments have been performed on Linux, while the plotting on Windows).

### Build and Execute

Follow these steps to build the project and run the executable:

```bash
cmake -B build
cmake --build build --config Release
.\build\Release\simulator.exe  # for Windows
./build/Release/simulator      # for Linux
```

### Obtain the Results

The output data required to generate the performance figures is saved in a file named `preliminary.csv`. Each of the following *Experiments* produces this file, which should be stored in an appropriate folder. For example, the data for Experiment 1a can be stored in a folder `belief`, from which then it is possible to plot the results.

- **Experiment 1a and 1b** (Figure 1): In the `main.cpp` file, uncomment the line following *Experiment 1 - extended seed selection* and comment the line following *Experiment 2 - reduced seed set*, to select the appropriate seed list. Similarly, choose the correct `alg_vec`: uncomment first *Experiment 1a - B-ColME* to produce Figure 1a, while commenting out the other lines. Then compile and execute. Save the `preliminary.csv` file in a subfolder (`belief`). Perform a similar procedure to produce Figure 1b, uncommenting *Experiment 1b - C-ColME* and commenting out the other two options. Again, compile, execute and save the `preliminary.csv` file in a subfolder (`consensus`).

- **Experiment 2** (Figure 2): In the `main.cpp` file now select the *Experiment 2 - reduced seed set* and choose the *Experiment 2 - Algorithms Comparison* to what concerns the choice of `alg_vec`. Run and execute, then save the `preliminary.csv` in a subfolder (`compare`).

Similarly, the setting can be adapted to conduct the additional experiments in the Supplementary Material.

### Plot the Results

Use the `requirements.txt` to install the required dependencies. The plotting has been performed with Python (3.9.13) over Windows. To obtain Figure 1a and 1b, run:

```bash
python plot_avg.py -l --no_title -s belief
```

and

```bash
python plot_avg.py -l --no_title -s consensus
```

Lastly, to obtain Figure 2 containing the comparison of the algorithms, run:

```bash
python plot_compare.py -l --no_title -s compare
```
