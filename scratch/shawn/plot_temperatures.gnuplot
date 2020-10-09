#! /usr/bin/gnuplot
# call like
# gnuplot -c plot_temperatures.gnuplot DATAFILE
datafile=ARG1
datafile_relpath=system(sprintf("basename %s",datafile))

set yrange [0:120]
set xlabel 'Time'
set ylabel 'Temperature (C)'
set grid

set timefmt "%s"
#set format x "%m/%d/%Y %H:%M:%S"
set format x "%H:%M:%S"
set xdata time

plot datafile u 2:3 title 'fpga\_temp', datafile u 2:7 title 'cc\_temp', datafile u 2:12 title 'atca\_temp\_fpga', datafile u 2:13 title 'atca\_temp\_rtm', datafile u 2:14 title 'atca\_temp\_amc0', datafile u 2:15 title 'atca\_temp\_amc2', datafile u 2:16 title 'atca\_jct\_temp\_fpga', datafile u 2:17 title 'regulator\_iout', datafile u 2:18 title 'regulator\_temp1', datafile u 2:19 title 'regulator\_temp2'

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
