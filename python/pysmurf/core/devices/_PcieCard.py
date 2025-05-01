#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : SMuRF PCIe Card
#-----------------------------------------------------------------------------
# File       : _PcieCard.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF PCIe card device.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
from pathlib import Path
import socket

import pyrogue

# Exit with a error message
def exit_message(message):
    print(message)
    print("")
    exit()

class PcieDev():
    """
    Class to setup each PCIe device

    This class contains wrapper to facilitate the process of setting
    up each PCIe device independently by the PcieCard class.

    This class must be used in a 'with' block in order to ensure that the
    RSSI connection is close correctly during exit even in the case of an
    exception condition.

    """
    def __init__(self, dev, name, description):
        import rogue.hardware.axi
        import SmurfKcu1500RssiOffload as fpga
        self._root = pyrogue.Root(name=name,description=description, serverPort=None, pollEn='False',initRead='True')
        self._memMap = rogue.hardware.axi.AxiMemMap(dev)
        self._root.add(fpga.Core(memBase=self._memMap))
        self._root.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._root.stop()

    def get_id(self):
        """
        Get the Device ID
        """
        return int(self._root.Core.AxiPcieCore.AxiVersion.DeviceId.get())

    def get_local_mac(self, lane):
        """
        Get the local MAC address for the specified Ethernet lane.
        """
        return self._root.Core.EthPhyGrp.EthConfig[lane].LocalMac.get()

    def set_local_mac(self, lane, mac):
        """
        Set the local MAC address for the specified Ethernet lane.
        """
        return self._root.Core.EthPhyGrp.EthConfig[lane].LocalMac.set(mac)

    def get_local_ip(self, lane):
        """
        Get the local IP address for the specified Ethernet lane.
        """
        return self._root.Core.EthPhyGrp.EthConfig[lane].LocalIp.get()

    def set_local_ip(self, lane, ip):
        """
        Set the local IP address for the specified Ethernet lane.
        """
        return self._root.Core.EthPhyGrp.EthConfig[lane].LocalIp.set(ip)

    def get_remote_ip(self, lane, client):
        """
        Get the remote IP address for the specified Ethernet lane.
        """
        return self._root.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[client].get()

    def set_remote_ip(self, lane, client, ip):
        """
        Set the remote IP address for the specified Ethernet lane.
        """
        return self._root.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[client].set(ip)

    def open_lane(self, lane, ip):
        """
        Open the RSSI connection on the specified lane, using the specified IP address.
        """
        print(f"    Opening PCIe RSSI lane {lane}")
        self._root.Core.UdpGrp.UdpConfig[lane].EnKeepAlive.set(1)
        self._root.Core.UdpGrp.UdpConfig[lane].KeepAliveConfig.set(0x2E90EDD0)  # 5 seconds
        self._root.Core.UdpGrp.RssiClient[lane].OpenConn.set(1)
        self._root.Core.UdpGrp.RssiClient[lane].CloseConn.set(0)
        self._root.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[0].set(ip)
        self._root.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[1].set(ip)
        self._root.Core.UdpGrp.UdpEngine[lane].ClientRemotePort[0].set(8198)
        self._root.Core.UdpGrp.UdpEngine[lane].ClientRemotePort[1].set(8195)

        # Print register status after setting them
        self.__print_lane_registers(lane)

    def close_lane(self, lane):
        """
        Close the RSSI connection on the specified Ethernet lane.
        """
        print(f"    Closing PCIe RSSI lane {lane}")
        self._root.Core.UdpGrp.UdpConfig[lane].KeepAliveConfig.set(0)
        self._root.Core.UdpGrp.RssiClient[lane].OpenConn.set(0)
        self._root.Core.UdpGrp.RssiClient[lane].CloseConn.set(1)
        self._root.Core.UdpGrp.UdpEngine[lane].ClientRemotePort[0].set(0)
        self._root.Core.UdpGrp.UdpEngine[lane].ClientRemotePort[1].set(0)

        # Print register status after setting them
        self.__print_lane_registers(lane)

    def __print_lane_registers(self, lane):
        """
        Print the register for the specified Ethernet lane.
        """
        print("      PCIe register status:")
        print(
            f"      Core.UdpGrp.UdpConfig[{lane}].EnKeepAlive         = " +
            f"{self._root.Core.UdpGrp.UdpConfig[lane].EnKeepAlive.get()}")
        print(
            f"      Core.UdpGrp.UdpConfig[{lane}].KeepAliveConfig     = 0x" +
            f"{self._root.Core.UdpGrp.UdpConfig[lane].KeepAliveConfig.get():02X}")
        print(
            f"      Core.UdpGrp.RssiClient[{lane}].OpenConn           = " +
            f"{self._root.Core.UdpGrp.RssiClient[lane].OpenConn.get()}")
        print(
            f"      Core.UdpGrp.RssiClient[{lane}].CloseConn          = " +
            f"{self._root.Core.UdpGrp.RssiClient[lane].CloseConn.get()}")
        print(
            f"      Core.UdpGrp.UdpEngine[{lane}].ClientRemotePort[0] = " +
            f"{self._root.Core.UdpGrp.UdpEngine[lane].ClientRemotePort[0].get()}")
        print(
            f"      Core.UdpGrp.UdpEngine[{lane}].ClientRemoteIp[0]   = " +
            f"{self._root.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[0].get()}")
        print(
            f"      Core.UdpGrp.UdpEngine[{lane}].ClientRemotePort[1] = " +
            f"{self._root.Core.UdpGrp.UdpEngine[lane].ClientRemotePort[1].get()}")
        print(
            f"      Core.UdpGrp.UdpEngine[{lane}].ClientRemoteIp[1]   = " +
            f"{self._root.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[1].get()}")
        print(
            f"      Core.EthPhyGrp.EthConfig[{lane}].LocalMac         = " +
            f"{self._root.Core.EthPhyGrp.EthConfig[lane].LocalMac.get()}")
        print(
            f"      Core.EthPhyGrp.EthConfig[{lane}].LocalIp          = " +
            f"{self._root.Core.EthPhyGrp.EthConfig[lane].LocalIp.get()}")
        print("")

    def print_version(self):
        """
        Print the PCIe device firmware information.
        """
        print("  ==============================================================")
        print(f"                   {self._root.description}")
        print("  ==============================================================")
        print("    FW Version      : 0x" +
              f"{self._root.Core.AxiPcieCore.AxiVersion.FpgaVersion.get():08X}")
        print("    FW GitHash      : 0x" +
              f"{self._root.Core.AxiPcieCore.AxiVersion.GitHash.get():040X}")
        print("    FW image name   : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.ImageName.get()}")
        print("    FW build env    : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.BuildEnv.get()}")
        print("    FW build server : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.BuildServer.get()}")
        print("    FW build date   : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.BuildDate.get()}")
        print("    FW builder      : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.Builder.get()}")
        print("    Up time         : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.UpTime.get()}")
        print("    Xilinx DNA ID   : 0x" +
              f"{self._root.Core.AxiPcieCore.AxiVersion.DeviceDna.get():032X}")
        print("    Device ID       : " +
              f"{self._root.Core.AxiPcieCore.AxiVersion.DeviceId.get()}")
        print("  ==============================================================")
        print("")

class PcieCard():
    """
    Class to setup the PCIe card devices.

    This class takes care of setting up both PCIe card devices
    according to the communication type used.

    If the PCIe card is present in the system:

    * All the RSSI connection lanes which point to the target IP
      address will be closed.
    * If PCIe communication type is used:
      * Verify that the DeviceId are correct for the RSSI (ID = 0) and the DATA (ID = 1) devices.
      * The RSSI connection is open in the specific lane. Also, when the the server is closed, the RSSI connection is closed.

    If the PCIe card is not present:

    * If PCIe communication type is used, the program is terminated.
    * If ETH communication type is used, then this class does not do
      anything.

    This class must be used in a 'with' block in order to ensure that
    the RSSI connection is close correctly during exit even in the
    case of an exception condition.
    """

    def __init__(self, comm_type, lane, ip_addr, dev_rssi, dev_data):

        print("Setting up the RSSI PCIe card...")

        # Get system status:

        # Check if the PCIe card for RSSI is present in the system
        if Path(dev_rssi).exists():
            self._pcie_rssi_present = True
            self._pcie_rssi = PcieDev(dev=dev_rssi, name='pcie_rssi', description='PCIe for RSSI')
        else:
            self._pcie_rssi_present = False

        # Check if the PCIe card for DATA is present in the system
        if Path(dev_data).exists():
            self._pcie_data_present = True
            self._pcie_data = PcieDev(dev=dev_data, name='pcie_data', description='PCIe for DATA')
        else:
            self._pcie_data_present = False

        # Check if we use the PCIe for communication
        if 'pcie-' in comm_type:
            self._use_pcie = True
        else:
            self._use_pcie = False

        # We need the IP address when the PCIe card is present, but not in used too.
        # If the PCIe card is present, this value could be updated later.
        # We need to know the IP address so we can look for all RSSI lanes that point
        # to it and close their connections.
        self._ip_addr = ip_addr

        # Look for configuration errors:

        if self._use_pcie:
            # Check if we are trying to use PCIe communication without the PCIe
            # cards present in the system
            if not self._pcie_rssi_present:
                exit_message(f"  ERROR: PCIe device {dev_rssi} not present.")

            if not self._pcie_data_present:
                exit_message(f"  ERROR: PCIe device {dev_data} not present.")

            # Verify the lane number is valid
            if lane is None:
                exit_message("  ERROR: Must specify an RSSI lane number")

            if lane in range(0, 6):
                self._lane = lane
            else:
                exit_message("  ERROR: Invalid RSSI lane number. Must be between 0 and 5")

            # We should need to check that the IP address is defined when PCIe is present
            # and not in used, but that is enforce in the main function.

            # Not more configuration errors at this point

            # Prepare the PCIe (DATA)
            with self._pcie_data as pcie:
                # Verify that its DeviceID is correct
                dev_data_id = pcie.get_id()
                if dev_data_id != 1:
                    exit_message(
                        f"  ERROR: The DeviceId for the PCIe dev for DATA is {dev_data_id} instead "
                        "of 1. Choose the correct device.")

                # Print FW information
                pcie.print_version()


            # Prepare the PCIe (RSSI)
            with self._pcie_rssi as pcie:
                # Verify that its DeviceID is correct
                dev_rssi_id = pcie.get_id()
                if dev_rssi_id != 0:
                    exit_message(
                        f"  ERROR: The DeviceId for the PCIe dev for RSSI is {dev_rssi_id} instead "
                        "of 0. Choose the correct device.")

                # Print FW information
                pcie.print_version()

                # Verify if the PCIe card is configured with a MAC and IP address.
                # If not, load default values before it can be used.
                valid_local_mac_addr = True
                local_mac_addr = pcie.get_local_mac(lane=self._lane)
                if local_mac_addr == "00:00:00:00:00:00":
                    valid_local_mac_addr = False
                    pcie.set_local_mac(lane=self._lane, mac=f"08:00:56:00:45:5{lane}")
                    local_mac_addr = pcie.get_local_mac(lane=self._lane)

                valid_local_ip_addr = True
                local_ip_addr = pcie.get_local_ip(lane=self._lane)
                if local_ip_addr == "0.0.0.0":
                    valid_local_ip_addr = False
                    pcie.set_local_ip(lane=self._lane, ip=f"10.0.1.20{lane}")
                    local_ip_addr = pcie.get_local_ip(lane=self._lane)


                # If the IP was not defined, read the one from the register space.
                # Note: this could be the case only the PCIe is in use.
                if not ip_addr:
                    ip_addr = pcie.get_remote_ip(lane=self._lane, client=0)

                    # Check if the IP address read from the PCIe card is valid
                    try:
                        socket.inet_pton(socket.AF_INET, ip_addr)
                    except socket.error:
                        exit_message(
                            f"ERROR: IP Address read from the PCIe card: {ip_addr} is invalid.")

                # Update the IP address.
                # Note: when the PCIe card is not in use, the IP will be defined
                # by the user.
                self._ip_addr = ip_addr

        # Yes no functions for reporting status
        def yes_or_no(b):
            return ("Yes" if b else "No")

        def yes_or_no_addr(b):
            return ("Yes" if b else "No. A default address was loaded")

        # Print system configuration and status
        print("  - PCIe for RSSI present in the system    : " +
              f"{yes_or_no(self._pcie_rssi_present)}")
        print("  - PCIe for Data present in the system    : " +
              f"{yes_or_no(self._pcie_data_present)}")
        print("  - PCIe based communication selected      : " +
              f"{yes_or_no(self._use_pcie)}")

        # Show IP address and lane when the PCIe is in use
        if self._use_pcie:
            print("  - Valid MAC address                      : " +
                  f"{yes_or_no_addr(valid_local_mac_addr)}")
            print("  - Valid IP address                       : " +
                  f"{yes_or_no_addr(valid_local_ip_addr)}")
            print(f"  - Local MAC address:                     : {local_mac_addr}")
            print(f"  - Local IP address:                      : {local_ip_addr}")
            print(f"  - Using IP address                       : {self._ip_addr}")
            print(f"  - Using RSSI lane number                 : {self._lane}")
            print("")

        # When the PCIe card is not present we don't do anything

    def __enter__(self):
        # Check if the PCIe card is present. If not, do not do anything.
        if self._pcie_rssi_present:

            # Close all RSSI lanes that point to the target IP address
            self.__close_all_rssi()

            # Check if the PCIe card is used. If not, do not do anything.
            if self._use_pcie:

                # Open the RSSI lane
                with self._pcie_rssi as pcie:
                    pcie.open_lane(lane=self._lane, ip=self._ip_addr)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Check if the PCIe card is present. If not, do not do anything.
        if self._pcie_rssi_present:

            # Check if the PCIe card is used. If not, do not do anything.
            if self._use_pcie:

                # Close the RSSI lane before exit,
                with self._pcie_rssi as pcie:
                    pcie.close_lane(self._lane)

    def __close_all_rssi(self):
        """
        Close all lanes with the target IP address
        """

        # Check if the PCIe is present
        if self._pcie_rssi_present:
            with self._pcie_rssi as pcie:
                print(f"  * Looking for RSSI lanes pointing to {self._ip_addr}...")
                # Look for links with the target IP address, and close their RSSI connection
                for i in range(6):
                    if self._ip_addr == pcie.get_remote_ip(lane=i, client=0):
                        print(f"    Lane {i} points to it. Disabling it...")
                        pcie.close_lane(i)
                        print("")
                print("  Done!")
                print("")

    def restart_rssi(self):
        if self._pcie_rssi_present:
            if self._use_pcie:
                with self._pcie_rssi as pcie:
                    pcie.close_lane(self._lane)
                    pcie.open_lane(lane=self._lane, ip=self._ip_addr)
