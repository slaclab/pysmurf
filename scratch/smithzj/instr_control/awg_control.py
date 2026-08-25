#!/usr/bin/env python3
import sys 
import numpy as np
import serial 
from time import sleep 

# Function to send SCPI commands
def send_command(inst, command):
    inst.write((command + '\n').encode())
    sleep(0.1)

# Function to query the instrument
def query_instrument(inst, command, response_return=False):
    send_command(inst, command)
    response = inst.readline().decode().strip()
    print(response)
    if response_return == True: 
        return response
    else: return 