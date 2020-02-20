#! /usr/bin/gnuplot
# call like
# gnuplot -c plot_temperatures.gnuplot DATAFILE
datafile=ARG1
datafile_relpath=system(sprintf("basename %s",datafile))

#1  epics_root
#2  ctime
#3  fpga_temp
#4  fpgca_vccint
#5  fpgca_vccaux
#6  fpgca_vccbram
#7  cc_temp
#8  bay0_dac0_temp
#9  bay0_dac1_temp
#10 bay1_dac0_temp
#11 bay1_dac1_temp
#12 atca_temp_fpga
#13 atca_temp_rtm
#14 atca_temp_amc0
#15 atca_temp_amc2
#16 atca_jct_temp_fpga
#17 regulator_iout
#18 regulator_temp1
#19 regulator_temp2  

while 1 {
      set yrange [0:130]
      set xlabel 'Time'
      set ylabel 'Temperature (C)'
      set grid
      
      set timefmt "%s"
      #set format x "%m/%d/%Y %H:%M:%S"
      set format x "%H:%M:%S"
      set xdata time

      set title datafile_relpath noenhanced
      set key font ",6"
      set key left bottom
      set xtics font ", 6"
      show title
      set key top left

      #change size of tick labels
      set tics font ", 6"

      plot datafile u 2:3 title 'fpga\_temp', datafile u 2:7 title 'cc\_temp', datafile u 2:12 title 'atca\_temp\_fpga', datafile u 2:13 title 'atca\_temp\_rtm', datafile u 2:14 title 'atca\_temp\_amc0', datafile u 2:15 title 'atca\_temp\_amc2', datafile u 2:16 title 'atca\_jct\_temp\_fpga', datafile u 2:17 title 'regulator\_iout', datafile u 2:18 title 'regulator\_temp1', datafile u 2:19 title 'regulator\_temp2'

      pause 3
}
