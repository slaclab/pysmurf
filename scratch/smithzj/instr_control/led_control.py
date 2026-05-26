#!/usr/bin/env python3
import sys 
print("running...")
import numpy as np
print('imported numpy')
import time
print('imported time')
import pyvisa 
print('imported pyvisa!!')

rm = pyvisa.ResourceManager()
resources = rm.list_resources()
#led_driver = rm.open_resource('/dev/usbtmc0')

# Example: Query the device
#response = led_driver.query('*IDN?')
#print(response)
print(resources)
rm.close()