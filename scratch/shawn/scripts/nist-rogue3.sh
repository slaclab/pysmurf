# 2/18/20 : smurf-srv10
# Drop into rogue 3 configuration 
pwd=$PWD

rm -v /data/smurf_startup_cfg/smurf_startup.cfg
ln -s /home/cryo/docker/pysmurf/nist-cmb/pysmurf/cfg_files/nist/cmb/nist_cmb_smurf_startup.cfg /data/smurf_startup_cfg/smurf_startup.cfg

rm -v /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg
rm -v /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_hbOnlyBay0.cfg
ln -s /home/cryo/docker/pysmurf/nist-cmb/pysmurf/cfg_files/nist/cmb/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_lbOnlyBay0.cfg
ln -s /home/cryo/docker/pysmurf/nist-cmb/pysmurf/cfg_files/nist/cmb/experiment_nistcmb_srv10_dspv3_cc02-02_hbOnlyBay0.cfg /data/pysmurf_cfg/experiment_nistcmb_srv10_dspv3_cc02-02_hbOnlyBay0.cfg

cd /home/cryo/docker/pysmurf/nist-cmb/pysmurf/scratch/shawn/scripts/
sudo ./install.sh

rm /home/cryo/docker/smurf/current
ln -s /home/cryo/docker/smurf/dev_fw/R3.1.2 /home/cryo/docker/smurf/current

cd $pwd
