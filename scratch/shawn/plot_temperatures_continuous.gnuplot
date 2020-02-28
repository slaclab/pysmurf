#! /usr/bin/gnuplot
# call like
# gnuplot -c plot_temperatures.gnuplot DATAFILE SLOT
datafile=ARG1
slot=ARG2
datafile_relpath=system(sprintf("basename %s",datafile))
smurf_server=sprintf("smurf_server_s%s",slot)

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

      set title sprintf("%s %s",datafile_relpath,smurf_server) noenhanced
      set key font ",6"
      set key left bottom
      set xtics font ", 6"
      show title
      set key top left

      #change size of tick labels
      set tics font ", 6"

      plot datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):3 title 'fpga\_temp', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):7 title 'cc\_temp', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):12 title 'atca\_temp\_fpga', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):13 title 'atca\_temp\_rtm', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):14 title 'atca\_temp\_amc0', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):15 title 'atca\_temp\_amc2', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):16 title 'atca\_jct\_temp\_fpga', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):17 title 'regulator\_iout', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):18 title 'regulator\_temp1', datafile u (stringcolumn(1) eq smurf_server ? $2 : 1/0):19 title 'regulator\_temp2'

      pause 6
}
