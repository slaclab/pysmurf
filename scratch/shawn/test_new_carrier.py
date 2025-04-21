import os
import pysmurf
import numpy as np
import sys
import time
import glob

slot = S.slot_number

wait_after_setup_min=1
print('-> Waiting {} min after setup.'.format(wait_after_setup_min))
wait_after_setup_sec=wait_after_setup_min*60
time.sleep(wait_after_setup_sec)

print('-> Checking JESD status')
rx_valid     = [0, 0]
rx_valid_cnt = [0, 0]
tx_valid     = [0, 0]
tx_valid_cnt = [0, 0]
for bay in [0, 1]:
    rx_valid[bay] = (S.get_jesd_rx_data_valid(bay) == 0x3F3)
    tx_valid[bay] = (S.get_jesd_tx_data_valid(bay) == 0x3CF)
    for i in range(10):
        rx_valid_cnt[bay] = rx_valid_cnt[bay] + S.get_jesd_rx_status_valid_cnt(bay, i)
        tx_valid_cnt[bay] = tx_valid_cnt[bay] + S.get_jesd_tx_status_valid_cnt(bay, i)

    print(f'    Bay {bay} JESD Rx Valid {rx_valid[bay]}')
    print(f'    Bay {bay} JESD Tx Valid {tx_valid[bay]}')
    print(f'    Bay {bay} JESD Rx Valid Count {rx_valid_cnt[bay]}')
    print(f'    Bay {bay} JESD Tx Valid Count {tx_valid_cnt[bay]}')

input('-> Make sure JESDs are all valid and counts are 0 (press enter)...')
print(' ')

print(f'-> Checking full band response to confirm RF is properly configured on slot {slot}.')    
exec(open("/usr/local/src/pysmurf/scratch/shawn/full_band_response.py").read())
print('-> Done running full_band_response.py.')

input(f'-> Visually check the measured full band response on slot {slot} before continuing (press enter)...')
print(' ')

wait_after_full_band_minutes=15
print(f'Waiting {wait_after_full_band_minutes} minutes until next check')
time.sleep(wait_after_full_band_minutes*60)

print('-> Checking JESD status')
rx_valid     = [0, 0]
rx_valid_cnt = [0, 0]
tx_valid     = [0, 0]
tx_valid_cnt = [0, 0]
for bay in [0, 1]:
    rx_valid[bay] = (S.get_jesd_rx_data_valid(bay) == 0x3F3)
    tx_valid[bay] = (S.get_jesd_tx_data_valid(bay) == 0x3CF)
    for i in range(10):
        rx_valid_cnt[bay] = rx_valid_cnt[bay] + S.get_jesd_rx_status_valid_cnt(bay, i)
        tx_valid_cnt[bay] = tx_valid_cnt[bay] + S.get_jesd_tx_status_valid_cnt(bay, i)

    print(f'    Bay {bay} JESD Rx Valid {rx_valid[bay]}')
    print(f'    Bay {bay} JESD Tx Valid {tx_valid[bay]}')
    print(f'    Bay {bay} JESD Rx Valid Count {rx_valid_cnt[bay]}')
    print(f'    Bay {bay} JESD Tx Valid Count {tx_valid_cnt[bay]}')
input('-> Make sure JESDs are all valid and counts are 0 (press enter)...')
print('Test complete.  Copy the logfile and loopback test image to the hardware database:')
print(max(glob.iglob('/data/smurf_data/tmux_logs/tmux*log'), key=os.path.getctime))
print(max(glob.iglob('/data/smurf_data/*/*/plots/*full_band_resp_all.png'), key=os.path.getctime))
