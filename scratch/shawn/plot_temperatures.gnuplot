#! /usr/bin/gnuplot
# call like
# gnuplot -c plot_temperatures.gnuplot DATAFILE
datafile=ARG1
datafile_relpath=system(sprintf("basename %s",datafile))

set yrange [0:80]
set xlabel 'Time'
set ylabel 'Temperature (C)'
set grid

set timefmt "%s"
#set format x "%m/%d/%Y %H:%M:%S"
set format x "%H:%M:%S"
set xdata time

plot datafile u 1:2 title 'FPGA BTemp', datafile u 1:3 title 'FPGA JTemp', datafile u 1:4 title 'DAC0 temp', datafile u 1:5 title 'DAC1 temp', datafile u 1:6 title 'AxiSysMonUltraScale:Temperature', datafile u 1:10 title 'CC temp'
set title datafile_relpath noenhanced
set key font ",8"
set key left top
set xtics font ", 8"
show title

#change size of tick labels
set tics font ", 6"

set term png
pngname=system(sprintf("echo %s | sed s/.dat/.png/g",datafile_relpath))
set output pngname
replot
set term x11

