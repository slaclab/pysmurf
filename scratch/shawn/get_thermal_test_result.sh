echo "dat="$1
awk '!/endrestrictfandwell/ { line = $0 } /endrestrictfandwell/ { print line }' $1 | awk '{print "FPGA temperature @ 50% fan speed (fpga_temp): "$3}'
awk '!/endrestrictfandwell/ { line = $0 } /endrestrictfandwell/ { print line }' $1 | awk '{print "Regulator current @ 50% fan speed (regulator_iout) : "$17}'
awk '!/endrestrictfandwell/ { line = $0 } /endrestrictfandwell/ { print line }' $1 | awk '{print "Regulator temperature 1 @ 50% fan speed (regulator_temp1) : "$18}'
awk '!/endrestrictfandwell/ { line = $0 } /endrestrictfandwell/ { print line }' $1 | awk '{print "Regulator temperature 2 @ 50% fan speed (regulator_temp2) : "$19}'

