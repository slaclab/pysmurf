#! /usr/bin/gnuplot
# call like
# gnuplot -c plot_voltages.gnuplot DATAFILE
datafile=ARG1
datafile_relpath=system(sprintf("basename %s",datafile))

set yrange [0:2]
set xlabel 'Time'
set ylabel 'Voltage (V)'
set grid

set timefmt "%s"
#set format x "%m/%d/%Y %H:%M:%S"
set format x "%H:%M:%S"
set xdata time

plot datafile u 1:9 title 'FPGA VccInt', datafile u 1:10 title 'FPGA VccAux', datafile u 1:11 title 'FPGA VccBram', 0.825 title 'Minimum VccInt spec' dt 2 lw 2
set title datafile_relpath noenhanced
set key font ",12"
set key left bottom
set xtics font ", 12"
show title

#change size of tick labels
set tics font ", 12"

#set term png
set terminal pngcairo enhanced font "Times New Roman,12.0" size 1500,1100
pngname=system(sprintf("echo %s | sed s/.dat/.png/g",datafile_relpath))
set output pngname
replot
set term x11


