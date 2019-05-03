
set yrange [0:100]
set xlabel 'Time'
set ylabel 'Temperature (C)'
set grid

set timefmt "%s"
#set format x "%m/%d/%Y %H:%M:%S"
set format x "%H:%M:%S"
set xdata time

plot '1556909023_temp.dat' u 1:2 title 'FPGA BTemp', '1556909023_temp.dat' u 1:3 title 'FPGA JTemp', '1556909023_temp.dat' u 1:4 title 'DAC0 temp', '1556909023_temp.dat' u 1:5 title 'DAC1 temp', '1556909023_temp.dat' u 1:6 title 'AxiSysMonUltraScale:Temperature', '1556909023_temp.dat' u 1:10 title 'CC temp'
set title '1556909023 - 400 tones on in bands 2 & 3, DSPv3 (mitch\_4\_30)'

#change size of tick labels
set tics font ", 10"

set term png
set output "1556909023_temp.png"
replot
set term x11

