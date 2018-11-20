# pysmurf
The python control software for SMuRF. Includes scripts to do low level commands as well as higher level analysis.

# Install
The code should all work out of the box with one exception. To read the GCP data, you will need to compile a c function. This will definitely change in the future, but for now go into the util subdirectory and type

g++ -o extractdata extractdata.cpp