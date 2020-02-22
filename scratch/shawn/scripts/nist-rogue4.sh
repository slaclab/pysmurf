# updated 2/12/20 : smurf-srv10
# Drop into rogue 4 configuration
pwd=$PWD

### for now, just use the rogue3 cfg files, but need to move into rogue4
rm -v /data/smurf_startup_cfg/smurf_startup.cfg
ln -s /home/cryo/docker/pysmurf/R4-rc1/pysmurf/cfg_files/nist/cmb/nist_cmb_smurf_startup.cfg /data/smurf_startup_cfg/smurf_startup.cfg

rm -v /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg
rm -v /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_hbOnlyBay0.cfg
ln -s /home/cryo/docker/pysmurf/R4-rc1/pysmurf/cfg_files/nist/cmb/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg
ln -s /home/cryo/docker/pysmurf/R4-rc1/pysmurf/cfg_files/nist/cmb/experiment_nistcmb_srv10_dspv3_cc02-02_hbOnlyBay0.cfg /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_hbOnlyBay0.cfg

cd /home/cryo/docker/pysmurf/R4-rc1/pysmurf/scratch/shawn/scripts/
sudo ./install.sh

rm /home/cryo/docker/smurf/current
ln -s /home/cryo/docker/smurf/dev_fw/slotN/v4.0.0-rc18 /home/cryo/docker/smurf/current

cd $wd
