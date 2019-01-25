import numpy as np
import pysmurf
import time
import sys

## Idea for a new fast tuning from Mitch & Shawn; private
## convo here with some details;
## https://slacsmurf.slack.com/archives/D8C76GH8D/p1542151795056000

## Brazenly stolen from 
## https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees
def limit_phase_deg(phase,minphase=-180):
    newPhase=phase
    while newPhase<=minphase:
        newPhase+=360
    while newPhase>minphase+360:
        newPhase-=360
    return newPhase

#stolen from https://stackoverflow.com/questions/3160699/python-progress-bar
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    sys.stdout.flush()
    text = "\rPercent: [{0}]".format( "#"*block + "-"*(barLength-block)) + " %3.0f%%"%(progress*100)
    sys.stdout.write(text)

#for i in range(100):
#    update_progress(float((i+1)/100.))    
#    time.sleep(0.1)
band=2

## NEED TO CODE IN POWER, FREQUENCIES
## PROBABLY THOSE ARE COMING AS INPUTS THOUGH

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='test_epics',cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_fp28_smurfsrv04.cfg')

Npts=20
##sb64
#channels=[256,288,320]
##sb63
channels=S.which_on(band)

## for each channel that's been assigned to a frequency,
## set:
##  feedbackEnable=0
##  etaMag=1
##  etaPhase=0

# feedbackEnable=0
feedbackEnableArray=S.get_feedback_enable_array(band)
feedbackEnableArray[channels]=0
S.set_feedback_enable_array(band,feedbackEnableArray)

#etaMag=1
etaMagArray=S.get_eta_mag_array(band)
etaMagArray[channels]=1
S.set_eta_mag_array(band,etaMagArray)

#etaPhase=0
etaPhaseArray=S.get_eta_phase_array(band)
etaPhaseArray[channels]=0
S.set_eta_phase_array(band,etaPhaseArray)

print('-> Measuring in etaPhase=0 direction')
adc0=np.zeros(shape=(Npts,len(channels)))
for i in range(Npts):
    update_progress(float((i+1)/Npts))    
    adc0[i,:]=S.get_frequency_error_array(band)[channels]
print('\n')

# now etaPhase=90
etaPhaseArray[channels]=90
S.set_eta_phase_array(band,etaPhaseArray)

print('-> Measuring in etaPhase=90 direction')
adc90=np.zeros(shape=(Npts,len(channels)))
for i in range(Npts):
    update_progress(float((i+1)/Npts))    
    adc90[i,:]=S.get_frequency_error_array(band)[channels]
print('\n')

## done measuring frequencyError at 0 and 90deg

adc0Est=np.median(adc0,axis=0)
adc90Est=np.median(adc90,axis=0)

# the -pi here is to match the etaScan results we get
# from the old matlab etaEstimator scripts
inPhaseRad = np.arctan2(adc90Est,adc0Est)
quadraturePhaseRad=inPhaseRad+np.pi/2.

inPhaseDeg = inPhaseRad*180./np.pi;
quadraturePhaseDeg = quadraturePhaseRad*180./np.pi;

print('inPhaseDeg=',inPhaseDeg)
print('quadraturePhaseDeg=',quadraturePhaseDeg)

## now must compute etaMagScaled
#
delF=0.01 # default 10kHz
centerFrequencyArray=S.get_center_frequency_array(band)

### 
# frequencies to measure at to derive eta

#f+dF
centerFrequencyArrayPlusDelF = np.copy(centerFrequencyArray)
centerFrequencyArrayPlusDelF[channels]+=delF

#f-dF
centerFrequencyArrayMinusDelF = np.copy(centerFrequencyArray)
centerFrequencyArrayMinusDelF[channels]-=delF

###
etaPhaseArray[channels]=[limit_phase_deg(qpd) for qpd in quadraturePhaseDeg]
S.set_eta_phase_array(band,etaPhaseArray)

# measure Quadrature Response at f+delF
print('-> Measuring quadrature response at f+delF with calibrated etaPhase')
adcQPlusDelF=np.zeros(shape=(Npts,len(channels)))

S.set_center_frequency_array(band,centerFrequencyArrayPlusDelF)
for i in range(Npts):
    update_progress(float((i+1)/Npts))    
    adcQPlusDelF[i,:]=S.get_frequency_error_array(band)[channels]
print('\n')

# measure Quadrature response at f-delF
print('-> Measuring quadrature response at f-delF with calibrated etaPhase')
adcQMinusDelF=np.zeros(shape=(Npts,len(channels)))
S.set_center_frequency_array(band,centerFrequencyArrayMinusDelF)
for i in range(Npts):
    update_progress(float((i+1)/Npts))    
    adcQMinusDelF[i,:]=S.get_frequency_error_array(band)[channels]
print('\n')

adcQPlusDelFEst=np.median(adcQPlusDelF,axis=0)
adcQMinusDelFEst=np.median(adcQMinusDelF,axis=0)

# 
# compute etaMagScaled
digitizerFrequencyMHz=S.get_digitizer_frequency_mhz(band)
numberSubBands=S.get_number_sub_bands(band)
subBandHalfWidthMHz=(digitizerFrequencyMHz/numberSubBands)

# this is only an estimate of eta because we're assuming we've
# correctly oriented the resonance circle.
etaEst = (2*delF/(adcQPlusDelFEst-adcQMinusDelFEst))
etaMag = abs(etaEst)
etaScaled=etaMag/subBandHalfWidthMHz

# if etaEst is negative, add 180 deg to etaPhase
etaPhaseArray=S.get_eta_phase_array(band)
etaPhaseArray[channels]=[limit_phase_deg(eP+180) if eE<0 else eP for (eP,eE) in zip(etaPhaseArray[channels],etaEst)]
S.set_eta_phase_array(band,etaPhaseArray)

# program etaScaled 
## the 10x is empirical right now...for some reason 
## not agreeing yet with setupNotches_umux16
etaMagScaledArray=S.get_eta_mag_array(band)
etaMagScaledArray[channels]=etaScaled*10.
S.set_eta_mag_array(band,etaMagScaledArray)

### set centerFrequency back to resonator and start integral tracking

# set tones back on resonance
S.set_center_frequency_array(band,centerFrequencyArray)

# feedbackEnable=1
feedbackEnableArray[channels]=1
S.set_feedback_enable_array(band,feedbackEnableArray)
