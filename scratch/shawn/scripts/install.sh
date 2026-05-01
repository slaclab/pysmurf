installdir=/usr/local/src/smurf-server-scripts/docker_scripts/

# smurf_startup{,_functions} live under the top-level scripts/ directory
# (renamed from shawnhammer{,functions}.sh per issue #85).
for f in ../../../scripts/smurf_startup.sh ../../../scripts/smurf_startup_functions.sh
do
    fbase=$(basename $f)
    fstrip=${fbase%.*}
    rm -f ${installdir}/$fstrip
    ln -s `realpath $f` ${installdir}/$fstrip
done

for f in ping_carrier.sh switch_carrier.sh
do
    fstrip=${f%.*}
    rm -f ${installdir}/$fstrip
    ln -s `pwd`/$f ${installdir}/$fstrip
done
