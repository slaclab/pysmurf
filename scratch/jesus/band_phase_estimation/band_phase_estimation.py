import time
import epics
import matplotlib.pyplot as plt
import pysmurf
import pysmurf.client

# Definitions
slot=5
band=2
tone_freq=[5100, 5102.4]
delay_step = 0.01 # us
delay_num_steps = 50
config_file="/usr/local/src/pysmurf/cfg_files/stanford/experiment_fp30_cc02-03_lbOnlyBay0.cfg"
epics_prefix=f'smurf_server_s{slot}'

# Create the SMuRF control object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file, setup=True, make_logfile=False)

# Turn on the fixed tones
print(f'Turning on tones at {tone_freq} MHz, on band {band}')
S.band_off(band=band)
for f in tone_freq:
    S.set_fixed_tone(freq_mhz=f, tone_power=11)

# Plot the DAC output
S.read_dac_data(data_length=2**16, band=band, make_plot=True, show_plot=True)

# Setup the BandPhaseFeedback device
epics.caput(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:toneChannels', list(range(0, len(tone_freq))))
epics.caput(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:toneFrequencies', tone_freq)
epics.caput(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:Disable', False)

# Start the data stream
S.stream_data_on()
time.sleep(2)

# Check if rhe BandPhaseFeedback is ready
if not epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:Ready'):
    print('BandPhaseFeedback module is not configured correctly. Aborting.')
else:
    # Sweep the band delay and record the tau and theta
    
    ## Prepare list objects
    d = []
    tau = []
    theta = []
    phase = []
    
    ## Save the initial band delay value
    initial_delay = S.get_band_delay_us(band=band)
    print(f'Initial band delay = {initial_delay} us')
    
    ## Sweep the band delay, starting and the current value
    delay = initial_delay
    print(f'Starting loop. Delay step = {delay_step} us, number of steps = {delay_num_steps}')
    for i in range(delay_num_steps):
        d.append(delay)
        S.set_band_delay_us(band=band, val=delay)
        time.sleep(5)
        tau.append(epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:Tau')*1e6)
        theta.append(epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:Theta'))
        phase.append(epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:tonePhases'))
        delay = delay + delay_step
    
    ## Restore the initial band delay value
    S.set_band_delay_us(band=band, val=initial_delay)
    
    # Stop the data stream
    S.stream_data_off()
    
    # Plot results
    plt.figure("Phase Time Delay Estimation (tau)")
    plt.plot(d, tau)
    plt.grid()
    plt.xlabel('Band delay set (using set_band_delay_us) [us]')
    plt.ylabel('Phase time delay estimation (tau) [us]')
    
    plt.figure("Phase Offset Estimation (theta)")
    plt.plot(d, theta)
    plt.grid()
    plt.xlabel('Band delay set (using set_band_delay_us) [us]')
    plt.ylabel('Phase offset estimation (theta) [us]')

    plt.figure("Raw phase")
    plt.plot(d, phase)
    plt.grid()
    plt.xlabel('Band delay set (using set_band_delay_us) [us]')
    plt.ylabel('Raw phase [ADC counts]')
