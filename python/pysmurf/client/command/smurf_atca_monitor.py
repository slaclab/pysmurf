#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf command module - SmurfAtcaMonitorMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/command/smurf_atca_monitor.py
# Created    : 2019-07-22
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
"""Defines the :class:`SmurfAtcaMonitorMixin` class."""
from pysmurf.client.base import SmurfBase

class SmurfAtcaMonitorMixin(SmurfBase):
    """Mixin providing interface with the atca_monitor server.

    This Mixin provides the pysmurf interface to the atca_monitor
    registers.  The atca_monitor server is a Rogue application
    which uses IPMI to monitor information from the ATCA system
    [#atca_monitor]_.  The atca_monitor server must be
    running or all queries will timeout and return `None`.

    References
    ----------
    .. [#atca_monitor] https://github.com/slaclab/smurf-atca-monitor

    """

    _write_atca_monitor_state_reg = "Crate.SaveState"

    def write_atca_monitor_state(self, val, **kwargs):
        """Writes atca_monitor state to yml file.

        Writes all current ATCA monitor values to a yml file.

        Args
        ----
        val : str
           The path (including file name) to write the yml file to.

        """
        self._caput( self._write_atca_monitor_state_reg, val, **kwargs)

    _board_temp_fpga_reg = 'BoardTemp.FPGA'

    def get_board_temp_fpga(self, slot_number=None, **kwargs):
        r"""Returns the AMC carrier board temperature.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the AMC carrier is installed
            into.  If None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            AMC carrier board temperature in Celsius.  If None, either
            the EPICS query timed out or the atca_monitor server
            isn't running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        return self._caget(
            f'Crate.Sensors.Slots.{slot_number}.' +
            self._board_temp_fpga_reg,**kwargs)

    _board_temp_rtm_reg = 'BoardTemp.RTM'

    def get_board_temp_rtm( self, slot_number=None, **kwargs):
        r"""Returns the RTM board temperature.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the RTM is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            RTM board temperature in Celsius.  If None, either the
            EPICS query timed out or the atca_monitor server isn't
            running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        return self._caget(
            f'Crate.Sensors.Slots.{slot_number}.' +
            self._board_temp_rtm_reg,**kwargs)

    _junction_temp_fpga_reg = 'JunctionTemp.FPG'

    def get_junction_temp_fpga( self, slot_number=None, **kwargs):
        r"""Returns FPGA junction temperature.

        FPGA die temperature - probably from a sensor on the FPGA.  If
        you are looking at this, you probably should be looking at
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_fpga_temp`
        instead, which we think is more reliable.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the FPGA carrier is installed
            into.  If None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            FPGA junction temperature in Celsius.  If None, either the
            EPICS query timed out or the atca_monitor server isn't
            running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        return self._caget( f'Crate.Sensors.Slots.{slot_number}.' + self._junction_temp_fpga_reg,**kwargs)

    _board_temp_amc_reg = 'BoardTemp.AMC{}'

    def get_board_temp_amc(self, bay, slot_number=None, **kwargs):
        r"""Returns the AMC board temperature.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        slot_number : int or None, optional, default None
            The crate slot number that the AMC is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            AMC board temperature in Celsius.  If None, either the
            EPICS query timed out or the atca_monitor server isn't
            running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        # For some reason, the bay 0 AMC is at AMC[0] and the bay 1
        # AMC is at AMC[2], hence the bay*2.
        return self._caget( f'Crate.Sensors.Slots.{slot_number}.' + self._board_temp_amc_reg.format(bay*2),**kwargs)

    _amc_product_asset_tag_reg = 'Product_Asset_Tag'
    _amc_product_version_reg = 'Product_Version'

    def get_amc_sn(
            self, bay, slot_number=None,
            shelf_manager=None,
            use_shell=False,
            **kwargs):
        r"""Returns the SMuRF AMC base board serial number.

        The AMC serial number is the combination of its 'Product
        Version' and 'Product Asset Tag' from its FRU data.  An
        example (the production AMCs built for Simons Observatory) is
        'Product Version'=C03 and 'Product Asset Tag'=A01-11', which
        combine to make the full AMC serial number C03-A01-11.

        C03 refers to the hardware revision of the AMC base board.
        The A## refers to the specific AMC baseboard loading.  The two
        most common SMuRF AMC base board loadings are A01 and A02
        corresponding to low band (4-6 GHz) and high band (6-8 GHz)
        AMCs.  The final number in the full serial number is the
        unique id assigned to each AMC base board which shares the
        same hardware revision and loading.

        By default, will try to get the serial number by querying the
        ATCA monitor EPICS server.  If you're not running the ATCA
        monitor, you can still get the AMC serial number more slowly
        via the shell by providing use_shell=True.

        Typical SMuRF AMC assemblies are composed of two connected
        boards, an AMC base board and an AMC RF daughter board.  The
        serial number of an AMC RF daughter card in an AMC assembly
        cannot be obtained remotely; you must either know which card
        your AMC was assembled with (or you can ask SLAC which
        maintains a database with this information) or it should be
        labeled on the frontpanel of your AMC assembly.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        slot_number : int or None, optional, default None
            The crate slot number of the carrier that the AMC is
            installed into.  If None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        use_shell : bool, optional, default False
            If False, polls the ATCA monitor EPICs server ; if True,
            runs slower shell command to poll this attribute.  This
            will be slower but provides an alternative if user is not
            running the ATCA monitor as part of their workflow, or if
            the ATCA monitor is down.
        shelf_manager : str or None, optional, default None
            Only used if use_shell=True.  If None, defaults to
            the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems the default name of the shelf manager
            is 'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str or None
            AMC serial number for the requested bay *e.g.*
            'C03-A01-01'.  If None, either the EPICS query timed out
            or the atca_monitor server isn't running, or if running
            with use_shell=True, the shell command failed.  Also
            returns None if there's no AMC in the requested bay if
            use_shell=True or if use_shell=True and the shell command
            used to poll the AMC FRU fails.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if shelf_manager is None:
            shelf_manager=self.shelf_manager
        if use_shell:
            amc_fru_dict = self.get_fru_info(board='amc',
                                             bay=bay,
                                             slot_number=slot_number,
                                             shelf_manager=shelf_manager)
            if amc_fru_dict is not None and amc_fru_dict.keys()>={'Product Version', 'Product Asset Tag'}:
                return f'{amc_fru_dict["Product Version"]}-{amc_fru_dict["Product Asset Tag"]}'
            else:
                self.log('ERROR : AMC FRU information incomplete or missing "Product Version" and/or "Product Asset Tag" fields.  Returning None.',
                         self.LOG_ERROR)
                return None
        else:
            # For some reason, the bay 0 AMC is at AMC[0] and the bay 1
            # AMC is at AMC[2], hence the bay*2.
            atca_epics_path=f'Crate.Sensors.Slots.{slot_number}.' + f'AMCInfo.{bay*2}.'
            amc_product_asset_tag=self._caget(atca_epics_path +
                                              self._amc_product_asset_tag_reg, as_string=True,
                                              **kwargs)
            amc_product_version=self._caget(atca_epics_path +
                                            self._amc_product_version_reg, as_string=True,
                                            **kwargs)
            return f'{amc_product_version}-{amc_product_asset_tag}'

    _carrier_product_asset_tag_reg = 'asset_tag'
    _carrier_product_version_reg = 'version'

    def get_carrier_sn(
            self, slot_number=None,
            shelf_manager=None,
            use_shell=False,
            **kwargs):
        r"""Returns the SMuRF carrier serial number.

        The carrier serial number is the combination of its 'Product
        Version' and 'Product Asset Tag' from its FRU data.  An
        example (the production carriers built for Simons Observatory) is
        'Product Version'=C03 and 'Product Asset Tag'=A04-50', which
        combine to make the full AMC serial number C03-A04-50.

        C03 refers to the hardware revision of the carrier board.  The
        A## refers to the specific carrier board loading.  The final
        number in the full serial number is the unique id assigned to
        each carrier board which shares the same hardware revision and
        loading.

        By default, will try to get the serial number by querying the
        ATCA monitor EPICS server.  If you're not running the ATCA
        monitor, can still get the carrier serial number more slowly
        via the shell by providing use_shell=True.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the carrier is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        use_shell : bool, optional, default False
            If False, polls the ATCA monitor EPICs server ; if True,
            runs slower shell command to poll this attribute.  This
            will be slower but provides an alternative if user is not
            running the ATCA monitor as part of their workflow, or if
            the ATCA monitor is down.
        shelf_manager : str or None, optional, default None
            Only used if use_shell=True.  If None, defaults to
            the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems the default name of the shelf manager
            is 'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str or None
            Carrier serial number *e.g.* 'C03-A04-50'.  If None,
            either the EPICS query timed out or the atca_monitor
            server isn't running, or if running with use_shell=True,
            the shell command failed.  Also returns None if there's no
            carrier in the requested slot if use_shell=True or if
            use_shell=True and the shell command used to poll the
            carrier FRU fails.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if shelf_manager is None:
            shelf_manager=self.shelf_manager
        if use_shell:
            carrier_fru_dict = self.get_fru_info(board='carrier',
                                             slot_number=slot_number,
                                             shelf_manager=shelf_manager)
            if carrier_fru_dict is not None and carrier_fru_dict.keys()>={'Product Version', 'Product Asset Tag'}:
                carrier_product_version=f'{carrier_fru_dict["Product Version"]}'
                carrier_product_asset_tag=f'{carrier_fru_dict["Product Asset Tag"]}'
                # Carrier frus can be a little hit or miss ...
                carrier_product_version=carrier_product_version.replace('_','-')
                carrier_product_asset_tag=carrier_product_asset_tag.split('-')[-1]

                return f'{carrier_product_version}-{carrier_product_asset_tag}'

            else:
                self.log('ERROR : Carrier FRU information incomplete or missing "Product Version" and/or "Product Asset Tag" fields.  Returning None.',
                         self.LOG_ERROR)
                return None
        else:
            atca_epics_path=f'Crate.Sensors.Slots.{slot_number}.CarrierInfo.'
            carrier_product_asset_tag=self._caget(atca_epics_path +
                                                  self._carrier_product_asset_tag_reg, as_string=True,
                                                  **kwargs)
            carrier_product_version=self._caget(atca_epics_path +
                                            self._carrier_product_version_reg, as_string=True,
                                            **kwargs)

            # Carrier frus can be a little hit or miss ...
            carrier_product_version=carrier_product_version.replace('_','-')
            carrier_product_asset_tag=carrier_product_asset_tag.split('-')[-1]

            return f'{carrier_product_version}-{carrier_product_asset_tag}'

    _rtm_product_asset_tag_reg = 'asset_tag'
    _rtm_product_version_reg = 'version'

    def get_rtm_sn(
            self, slot_number=None,
            shelf_manager=None,
            use_shell=False,
            **kwargs):
        r"""Returns the SMuRF carrier serial number.

        The RTM serial number is the combination of its 'Product
        Version' and 'Product Asset Tag' from its FRU data.  An
        example (the production RTM built for Simons Observatory) is
        'Product Version'=C01 and 'Product Asset Tag'=A01-02', which
        combine to make the full RTM serial number C01-A01-02.

        C01 refers to the hardware revision of the RTM board.  The A##
        refers to the specific RTM board loading.  The final number in
        the full serial number is the unique id assigned to each RTM
        board which shares the same hardware revision and loading.

        By default, will try to get the serial number by querying the
        ATCA monitor EPICS server.  If you're not running the ATCA
        monitor, can still get the RTM serial number more slowly
        via the shell by providing use_shell=True.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the RTM is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        use_shell : bool, optional, default False
            If False, polls the ATCA monitor EPICs server ; if True,
            runs slower shell command to poll this attribute.  This
            will be slower but provides an alternative if user is not
            running the ATCA monitor as part of their workflow, or if
            the ATCA monitor is down.
        shelf_manager : str or None, optional, default None
            Only used if use_shell=True.  If None, defaults to
            the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems the default name of the shelf manager
            is 'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str or None
            RTM serial number *e.g.* 'C01-A01-02'.  If None, either
            the EPICS query timed out or the atca_monitor server isn't
            running, or if running with use_shell=True, the shell
            command failed.  Also returns None if there's no RTM
            in the requested slot if use_shell=True or if
            use_shell=True and the shell command used to poll the
            RTM FRU fails.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if shelf_manager is None:
            shelf_manager=self.shelf_manager
        if use_shell:
            rtm_fru_dict = self.get_fru_info(board='rtm',
                                             slot_number=slot_number,
                                             shelf_manager=shelf_manager)
            if rtm_fru_dict is not None and rtm_fru_dict.keys()>={'Product Version', 'Product Asset Tag'}:
                rtm_product_version=f'{rtm_fru_dict["Product Version"]}'
                rtm_product_asset_tag=f'{rtm_fru_dict["Product Asset Tag"]}'
                return f'{rtm_product_version}-{rtm_product_asset_tag}'

            else:
                self.log('ERROR : RTM FRU information incomplete or missing "Product Version" and/or "Product Asset Tag" fields.  Returning None.',
                         self.LOG_ERROR)
                return None
        else:
            atca_epics_path=f'Crate.Sensors.Slots.{slot_number}.RTMInfo.'
            rtm_product_asset_tag=self._caget(atca_epics_path +
                                              self._rtm_product_asset_tag_reg, as_string=True,
                                              **kwargs)
            rtm_product_version=self._caget(atca_epics_path +
                                            self._rtm_product_version_reg, as_string=True,
                                            **kwargs)
            return f'{rtm_product_version}-{rtm_product_asset_tag}'
