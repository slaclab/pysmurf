#! /usr/bin/gnuplot
# call like
# gnuplot -c plot_temperatures.gnuplot DATAFILE
datafile=ARG1
datafile_relpath=system(sprintf("basename %s",datafile))

set yrange [0:100]
set xlabel 'Time'
set ylabel 'Temperature (C)'
set grid

set timefmt "%s"
#set format x "%m/%d/%Y %H:%M:%S"
set format x "%H:%M:%S"
set xdata time

plot datafile u 1:2 title 'FPGA BTemp', datafile u 1:3 title 'FPGA JTemp', datafile u 1:4 title 'Bay 0 DAC0 temp', datafile u 1:5 title 'Bay 0 DAC1 temp', datafile u 1:6 title 'Bay 1 DAC0 temp', datafile u 1:7 title 'Bay 1 DAC1 temp', datafile u 1:8 title 'AxiSysMonUltraScale:Temperature'
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


