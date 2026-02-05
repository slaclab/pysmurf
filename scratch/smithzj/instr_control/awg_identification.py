#!/usr/bin/env python3
import sys 
print("running...")
import numpy as np
print('imported numpy')
import serial 
print('imported serial')
import time
print('imported time')
# Open the serial port
ser = serial.Serial(
    port='/dev/ttyUSB0',       # Replace with your COM port
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)
print("imported ttyUSB0!!")
# Function to send SCPI commands
def send_command(command):
    ser.write((command + '\n').encode())
    time.sleep(0.1)

# Function to query the instrument
def query_instrument(command):
    send_command(command)
    response = ser.readline().decode().strip()
    return response

resp = query_instrument('*IDN?')
print(resp)

# Define the arbitrary waveform data
sampling_rate = 100000  # 100 kHz sampling rate
total_duration_ms = 50  # 50 ms total duration
total_samples = int(sampling_rate * total_duration_ms / 1000)  # Total samples = 5000
segment_duration_ms = 10  # 10 ms duration between segments
segment_samples = int(sampling_rate * segment_duration_ms / 1000)  # Samples per segment
t_us = 1000  # Duration of each square wave in microseconds (1 ms)

# Define the amplitudes for each of the 5 segments
amplitudes = [1, 0.5, -0.5, -1, 0.2]  # Example amplitudes for the 5 segments

# Create the arbitrary waveform data
waveform_data = []

for amplitude in amplitudes:
    # Calculate the number of points for each square wave (high and low) given t_us
    t_samples = int(sampling_rate * (t_us / 1e6))  # Convert t in microseconds to number of samples

    # Create one segment with alternating high and low values
    high_segment = [amplitude] * t_samples  # High segment for duration t
    low_segment = [-amplitude] * t_samples  # Low segment for duration t

    # Combine to form a complete segment
    segment = high_segment + low_segment
    segment = segment[:segment_samples]  # Ensure the segment is the correct length

    # Add the segment to the overall waveform
    waveform_data.extend(segment)

# Ensure total waveform length is 5000 samples
waveform_data = waveform_data[:total_samples] + [0] * (total_samples - len(waveform_data))

# Convert the waveform data to a comma-separated string
waveform_string = ','.join(f'{point:.5f}' for point in waveform_data)

try:
    # Reset and configure the instrument
    send_command('*RST')
    send_command('FUNC:ARB')
    resp = query_instrument('FUNC?')
    print(resp)
    resp = query_instrument('SYSTem:ERRor?')
    print(resp)
    #send_command(f'DATA:ARB VOLATILE,{waveform_string}')  # Upload arbitrary waveform
    #send_command('FUNC ARB')  # Set to arbitrary waveform mode
    #send_command('FUNC:ARB VOLATILE')  # Select the uploaded waveform
    #send_command('FREQ 20')  # Set frequency to 20 Hz
    #send_command('VOLT 1')  # Set amplitude to 1 V
    #send_command('OUTP ON')  # Enable the output

    # Query the instrument for confirmation
    response = query_instrument('*IDN?')
    print(f'Instrument ID: {response}')

finally:
    ser.close()  # Close the serial connection
