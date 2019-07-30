installdir=/usr/local/src/smurf-server-scripts/docker_scripts/

for f in shawnhammer.sh ping_carrier.sh switch_carrier.sh
do
    fstrip=${f%.*}
    rm ${installdir}/$fstrip
    ln -s `pwd`/$f ${installdir}/$fstrip
done
