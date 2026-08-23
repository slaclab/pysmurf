installdir=/usr/local/src/smurf-server-scripts/docker_scripts/

for f in smurfhammer.sh ping_carrier.sh switch_carrier.sh smurfhammerfunctions.sh
do
    fstrip=${f%.*}
    rm -f ${installdir}/$fstrip
    ln -s `pwd`/$f ${installdir}/$fstrip
done

# Remove old shawnhammer symlinks if present
rm -f ${installdir}/shawnhammer ${installdir}/shawnhammerfunctions
