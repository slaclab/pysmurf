import time
import epics
import pysmurf
import pysmurf.client

# Definitions
slot=5
band=2
tone_freq=[5100, 5102.4]
data_file='/data/jesus/band_estimation.dat'
config_file="/usr/local/src/pysmurf/cfg_files/stanford/experiment_fp30_cc02-03_lbOnlyBay0.cfg"
epics_prefix=f'smurf_server_s{slot}'

# Create the SMuRF control object
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file, setup=False, make_logfile=False)

# Turn on the fixed tones
print(f'Turning on tones at {tone_freq} MHz, on band {band}')
S.band_off(band=band)
for f in tone_freq:
    S.set_fixed_tone(freq_mhz=f, tone_power=11)

# Plot the DAC output
#S.read_dac_data(data_length=2**16, band=band, make_plot=True, show_plot=True)

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

    # Write data to file
    while True:
        with open(data_file, 'a') as f:
            t = time.time()
            tau = epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:Tau')
            theta = epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:Theta')
            phase = epics.caget(f'{epics_prefix}:AMCc:SmurfProcessor:BandPhaseFeedback[{band}]:tonePhases')
            f.write(f'{t} {tau} {theta} {phase}\n')

        time.sleep(60)

# Stop the data stream
S.stream_data_off()
