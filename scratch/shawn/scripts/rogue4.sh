# updated 2/12/20 : smurf-srv12
# Drop into rogue 4 configuration

pwd=$PWD

rm -v /data/smurf_startup_cfg/smurf_startup.cfg
ln -s /home/cryo/docker/pysmurf/R4-rc1/pysmurf/cfg_files/stanford/fp30_smurf_startup.cfg /data/smurf_startup_cfg/smurf_startup.cfg

rm -v /data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg
ln -s /home/cryo/docker/pysmurf/R4-rc1/pysmurf/cfg_files/stanford/experiment_fp30_cc02-03_lbOnlyBay0.cfg /data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg

cd /home/cryo/docker/pysmurf/R4-rc1/pysmurf/scratch/shawn/scripts/
sudo ./install.sh

rm /home/cryo/docker/smurf/current

ln -s /home/cryo/docker/smurf/dev_fw/slotN/v4.0.0-rc21/ /home/cryo/docker/smurf/current

cd $pwd
