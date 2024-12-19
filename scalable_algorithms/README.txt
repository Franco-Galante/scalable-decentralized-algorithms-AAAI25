To build and execute the project cmake is needed.
To obtain the data (csv) to later be processed with python to obtain the perfor-
mance of B-COLME and C-COLME (Figure 1) and the algorithm comparison (Figure 2),
type the following three lines:

cmake -B build
cmake --build build --config Release
.\build\Release\simulator.exe

and then use Python to plot the results, Figure 1:
python .\plot_avg.py -l --no_title

and Figure 2
python .\plot_comparison.py -l --no_title