open_project stann_examples_tb
add_files stann_examples.cpp
add_files -tb testbench/testbench.cpp
open_solution "solution1" -flow_target vivado
csim_design
exit

