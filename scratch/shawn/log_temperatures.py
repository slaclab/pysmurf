import pysmurf
import time
import subprocess

time_btw_meas=5 # sec

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='test_epics2',cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_fp29_smurfsrv04.cfg')

of=open('%s_temp.dat'%S.get_timestamp(),'w+')
hdr='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}{0[4]:<15}{0[5]:<15}{0[6]:<15}{0[7]:<15}{0[8]:<15}{0[9]:<15}\n'.format(['ctime','BTemp','JTemp','dac0_temp','dac1_temp','fpga_temp','fpgca_vccint','fpgca_vccaux','fpgca_vccbram','cc_temp'])
of.write(hdr)
of.flush()
print(hdr.rstrip())

#Mitch asks for us to add these
#[cryo@smurf-srv04 shawn]$ amcc_dump --fpga shm-smrf-sp01/2
#================================================================================
#| FPGA | Pre | Ena | Vok | BTemp | JTemp |
#================================================================================
#| shm-smrf-sp01/2                                                              |
#--------------------------------------------------------------------------------
#|  CEN |  Y  |  Y  |  Y  |  51C  |  87C  |                                     |
#--------------------------------------------------------------------------------
#================================================================================

while True:

    #amcc_dump --fpga
    x=subprocess.check_output(['amcc_dump', '--fpga','shm-smrf-sp01/2'])
    # parse (stupidly)
    x=list(filter(None,[x.strip() for x in list(filter(None,str(x).split('\\n')[5].split('|')))]))
    JTemp=x[5].rstrip('C')
    BTemp=x[4].rstrip('C')

    dac0_temp=S.get_dac_temp(0,0)
    dac1_temp=S.get_dac_temp(0,1)

    ctime=S.get_timestamp()

    fpga_temp=S.get_fpga_temp()
    fpgca_vccint=S.get_fpga_vccint()
    fpgca_vccaux=S.get_fpga_vccaux()
    fpgca_vccbram=S.get_fpga_vccbram()

    cc_temp=S.get_cryo_card_temp()

    data='{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}{0[4]:<15}{0[5]:<15}{0[6]:<15}{0[7]:<15}{0[8]:<15}{0[9]:<15}\n'.format([str(ctime),BTemp,JTemp,str(dac0_temp),str(dac1_temp),'%0.4f'%fpga_temp,'%0.4f'%fpgca_vccint,'%0.4f'%fpgca_vccaux,'%0.4f'%fpgca_vccbram,'%0.4f'%cc_temp])
    of.write(data)
    of.flush()
    print(data.rstrip())
    time.sleep(15)

of.close()
    
