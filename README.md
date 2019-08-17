# BioMETA

BioMETA: A multiple specification parameter estimation system for stochastic biochemical models

-----------------------------------------
Installation Instructions
-----------------------------------------

Prerequisite: Anaconda.

1) Download Anaconda according to your platform: 
https://www.anaconda.com/distribution/

2) Install Anaconda (example for Linux):
> bash Anaconda-latest-Linux-x86_64.sh

3) Initialize conda:
> conda init

Installation:
1) Download the git repository of the code and navigate inside the "parameter-synth" folder.

2) Use the makefile to setup a new anaconda environment by typing the following command:
> make install

3) Run the parameter estimation process by typing the following command:
> make run spec=<spec_filename> n=<no_of_runs>

This will run 'n' number of parameter estimation processes in the background for the given 'spec_filename'. The output on the terminal will show the time taken by each estimation process to find the parameters or to exit the process in case no parameters are found. Estimated set of parameters can be found inside the 'outputs/' folder in the form of a text file named 'estimated_parameters.txt'. (NOTE: Default value for 'n' is 20)

Examples (Fceri Model):
> make run spec=fceri_1 n=15      #(estimates parameters of Fceri model agaisnt Property 1)

> make run spec=fceri_2 n=20      #(estimates parameters of Fceri model agaisnt Property 2)

> make run spec=fceri_1_2_3 n=20  #(estimates parameters of Fceri model agaisnt Property 1, 2 and 3)


Examples (T-cell Model):
> make run spec=t-cell_1 n=10      #(estimates parameters of T-cell model agaisnt Property 1)

> make run spec=t-cell_1_2 n=30      #(estimates parameters of T-cell model agaisnt Property 1 and 2)

> make run spec=t-cell_1_2_3 n=40  #(estimates parameters of T-cell model agaisnt Property 1, 2 and 3)

