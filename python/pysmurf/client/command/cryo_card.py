#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf command module - CryoCard class
#-----------------------------------------------------------------------------
# File       : pysmurf/command/cryo_card.py
# Created    : 2018-09-21
# Author     : Joe Frisch
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import time
import os

try:
    import epics
except ModuleNotFoundError:
    print("cryo_card.py - epics not found.")

def write_csv(filename, header, line):
    should_write_header = os.path.exists(filename)
    with open(filename, 'a+') as f:
        if not should_write_header:
            f.write(header+'\n')
        f.write(line+'\n')

class CryoCard():
    def __init__(self, readpv_in, writepv_in):
        self.readpv = readpv_in
        self.writepv = writepv_in
        self.fw_version_address = 0x0
        self.relay_address = 0x2
        self.hemt_bias_address = 0x3
        self.a50K_bias_address = 0x4
        self.temperature_address = 0x5
        self.cycle_count_address = 0x6  # used for testing
        self.ps_en_address = 0x7 # PS enable (HEMT: bit 0, 50k: bit 1)
        self.ac_dc_status_address = 0x8 # AC/DC mode status (bit 0: FRN_RLY, bit 1: FRP_RLY)
        self.optical_address = 0x0D # Optical transmitter. Bit 0: TX1, Bit 1: TX2
        self.adc_scale = 3.3/(1024.0 * 5)
        self.temperature_scale = 1/.028 # was 100
        self.temperature_offset =.25
        self.bias_scale = 1.0
        self.max_retries = 5 #number of re-tries waiting for response
        self.retry = 0 # counts nubmer of retries
        self.busy_retry = 0  # counts number of retries due to relay busy status

    def do_read(self, address):
        #need double write to make sure buffer is updated
        epics.caput(self.writepv, cmd_make(1, address, 0))
        for self.retry in range(0, self.max_retries):
            epics.caput(self.writepv, cmd_make(1, address, 0))
            data = epics.caget(self.readpv)
            addrrb = cmd_address(data)
            if (addrrb == address):
                return(data)
        return(0)

        return (epics.caget(self.readpv))

    def do_write(self, address, value):
        """Write the given value directly to the address on the PIC. Make sure
        you know if the value should be base-16, base-10, or base-2. There are
        higher abstractions that might be more useful for what you're trying to
        do.

        :param address the address on the PIC (e.g. 0x2)
        :returns the response from caput
        """
        return epics.caput(self.writepv, cmd_make(0, address, value))

    def write_relays(self, relay):  # relay is the bit partern to set
        epics.caput(self.writepv, cmd_make(0, self.relay_address, relay))
        time.sleep(0.1)
        epics.caput(self.writepv, cmd_make(0, self.relay_address, relay))

    def read_relays(self):
        for self.busy_retry in range(0, self.max_retries):
            data = self.do_read(self.relay_address)
            if ~(data & 0x80000):  # check that not moving
                return(data & 0x7FFFF)
                time.sleep(0.1) # wait for relays to move
        return(80000) # busy flag still set

    def set_relay_bit(self, bit, value):
        """
        Set the relay bit on the cryocard to either 0 or 1. Bits 0-15 set the
        actual TES relays, and bit 16 sets the AC/DC coupling mode. Use
        set_ac_dc_delay and get_ac_dc_relay for convenience.

        Args
        ----
        bit (int) : 0-16 inclusive.
        value (int) : 0 or 1.

        Returns
        -------
        None
        """

        assert (bit in range(17)), ('bit must be in [0,...,16]')
        assert (value in [0,1]), ('value must be either 0 or 1')
        currentRelay = self.read_relays()
        nextRelay = currentRelay & ~(1<<bit)
        nextRelay = nextRelay | (value<<bit)
        self.write_relays(nextRelay)

    def get_relay_bit(self, bit):
        """
        Get the n-th bit from the relay address on the PIC. Bits 0-15 are the
        actual TES relays, where 1 is delatch mode, and 0 is not delatch mode.
        Bit 16 is the status of the flux ramp coupling, either AC mode (1) or
        DC mode (0).

        Args
        ----
        bit (int) : Which bit to read. 0-16.

        Returns
        -------
        (int) : The value of the bit, either 0 or 1.
        """
        relays = self.read_relays()

        if relays & 1 << bit > 0:
            return 1

        return 0

    def set_ac_dc_relay(self, value):
        """
        Set the AC/DC relay bit. 1 is AC coupled, and 0 is DC coupled.

        Args
        ----
        value (int) : 0 or 1.

        Returns
        -------
        None
        """
        assert (value in [0,1]), ('value must be either 0 or 1')

        self.set_relay_bit(16, value)

    def read_ac_dc_relay_status(self):
        """
        Read the relay for the flux ramp coupling mode, which is either AC or DC.

        Args
        ----
        None

        Returns
        -------
        (int): 0 is DC coupled, 1 is AC coupled.
        """
        return self.get_relay_bit(16)

    def delatch_bit(self, bit): # bit is the pattern for the desired relay, eg 0x4 for 100
        current_relay = self.read_relays()
        set_relay = current_relay + bit
        self.write_relays(set_relay)
        time.sleep(0.1)
        self.write_relays(current_relay) # return to original state

    def read_hemt_bias(self):
        data = self.do_read(self.hemt_bias_address)
        return((data& 0xFFFFF) * self.bias_scale * self.adc_scale)

    def read_50k_bias(self):
        data = self.do_read(self.a50K_bias_address)
        return((data& 0xFFFFF) * self.bias_scale * self.adc_scale)

    def read_temperature(self):
        data = self.do_read(self.temperature_address)
        volts = (data & 0xFFFFF) * self.adc_scale
        return((volts - self.temperature_offset) * self.temperature_scale)

    def read_cycle_count(self):
        data = self.do_read(self.count_address)
        return( cmd_data(data))  # do we have the right addres

    def write_ps_en(self, enables):
        """
        Write the power supply enable signals.

        Args
        ----
        enables (int): 2-bit number to set the power supplies enables.
           Bit 0 set the enable for HEMT power supply.
           Bit 1 set the enable for 50k power supply.
           Bit set to 1 mean enable power supply.
           Bit set to 0 mean disable the power supply.

        Returns
        -------
        Nothing
        """
        epics.caput(self.writepv, cmd_make(0, self.ps_en_address, enables))

    def read_ps_en(self):
        """
        Read the power supply enable signals.

        Args
        ----
        None


        Returns
        -------
        enables (int): 2-bit number with the status of the power supplies enables.
           Bit 0 for the HEMT power supply.
           Bit 1 for the 50k power supply.
           Bit set to 1 means the power supply is enabled.
           Bit set to 0 means the power supply is disabled.
        """
        data = self.do_read(self.ps_en_address)
        return(cmd_data(data))

    def get_fw_version(self):
        """
        Return the firmware version string from the Cryocard PIC controller.
        This can be used to check communication with the PIC is OK, and crudely
        proxy if this cryocard is of type C04 or C02. Addendum, the firmware
        version is only avaiable at register address 0x00 in PIC firmware
        versions R1.1.0+.  All previous versions of the code will return
        0xABCDE in this register.

        Returns
        -------
        (string) : Firmware version string from the Cryocard.
        """
        data = cmd_data(self.do_read(self.fw_version_address))

        hexstr = f'{data:06x}'

        if data == 0xABCDE:
            print('Cryostat card PIC firmware version read returned\n'
                  '0xABCDE, which means the firmware version number\n'
                  'wasn\'t loaded into the register at address 0x0\n'
                  'for this firmware version.  The firmware version\n'
                  'should be available in firmware releases\n'
                  'R1.1.0+, so the current firmware likely predates\n'
                  'R1.1.0.  Returning None.\n')
            return None

        patch = int(hexstr[-2:],16)
        minor = int(hexstr[-4:-2],16)
        major = int(hexstr[-6:-4],16)

        return(f'R{major}.{minor}.{patch}')

    def read_optical(self):
        """
        Read the state of both optical transmitters..

        Args
        ----
        None

        Returns
        -------
        (int): 2-bit number with the readback relay status
            Bit 0: If TX1 is enabled
            Bit 1: If Tx2 is enabled
        """
        return self.do_read(self.optical_address)

    def write_optical(self, value):
        """
        """
        return self.do_write(self.optical_address, value)

# low level data conversion

def cmd_read(data):  # checks for a read bit set in data
    return( (data & 0x80000000) != 0)

def cmd_address(data): # returns address data
    return((data & 0x7FFF0000) >> 20)

def cmd_data(data):  # returns data
    return(data & 0xFFFFF)

def cmd_make(read, address, data):
    return((read << 31) | ((address << 20) & 0x7FFF00000) | (data & 0xFFFFF))
