"""
JESD204B Comprehensive Link Diagnostic and Adaptive Tuning Script
for SMuRF C03 Production Systems.

Based on:
  - JESD_Link_Tuning_Guide.md (full register map + tuning procedures)
  - cryo-det firmware: surf/protocols/jesd204b/ (JesdRxLane, JesdSyncFsmRx,
    Jesd204bRx, JesdSysrefMon, JesdRx.yaml, JesdTx.yaml)
  - JESD204B-Survival-Guide.pdf (ADI MS-2448 troubleshooting procedures)
  - defaults_c03_lb_lb.yml (production C03 default register settings)
  - pysmurf JesdWatchdog.py (recovery sequence reference)

Runtime: Expect 2-6 hours depending on findings (configurable).

Usage:
    From a pysmurf-enabled environment:
        %run jesd_diagnose_and_tune.py
        diag = JesdDiagTune(S, bay=0)
        results = diag.run()

    Or with custom timing:
        diag = JesdDiagTune(S, bay=0, diag_dwell=60, sweep_dwell=120)
        results = diag.run()
"""

import time
import sys
import json
import argparse
import numpy as np
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from copy import deepcopy

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# Constants from firmware and defaults_c03_lb_lb.yml
# =============================================================================

# C03 default lane enable masks
DEFAULT_RX_ENABLE = 0x3F3    # Lanes 0,1,4,5,6,7,8,9
DEFAULT_TX_ENABLE = 0x3CF    # Lanes 0,1,2,3,6,7

# Which RX lanes are enabled (bit positions of DEFAULT_RX_ENABLE)
ENABLED_RX_LANES = [0, 1, 4, 5, 6, 7, 8, 9]
ENABLED_TX_LANES = [0, 1, 2, 3, 6, 7]

# ADC lane mapping: physical RX lanes -> ADC device/channel
# ADC[0] uses lanes 0,1 (SEL_EMP_LANE0, SEL_EMP_LANE2 are the active outputs)
# ADC[1] uses lanes 4,5,6,7,8,9
ADC_LANE_MAP = {
    0: {'adc': 0, 'desc': 'ADC[0] Lane A'},
    1: {'adc': 0, 'desc': 'ADC[0] Lane B'},
    4: {'adc': 1, 'desc': 'ADC[1] Lane A'},
    5: {'adc': 1, 'desc': 'ADC[1] Lane B'},
    6: {'adc': 1, 'desc': 'ADC[1] Lane C'},
    7: {'adc': 1, 'desc': 'ADC[1] Lane D'},
    8: {'adc': 0, 'desc': 'ADC[0] Lane C'},
    9: {'adc': 0, 'desc': 'ADC[0] Lane D'},
}

# LinkErrMask bit definitions (from JesdRxLane.vhd:263)
# s_linkErrVec <= s_positionErr & s_bufOvf & s_bufUnf &
#                 uOr(dispErr) & uOr(decErr) & s_alignErr
LINK_ERR_BITS = OrderedDict([
    (0, 'alignErr'),
    (1, 'decErr'),
    (2, 'dispErr'),
    (3, 'bufUnf'),
    (4, 'bufOvf'),
    (5, 'positionErr'),
])

# Error classification by root cause domain
ERR_CLASS = {
    'signal_integrity': ['dispErr', 'decErr'],
    'timing_buffer': ['bufUnf', 'bufOvf'],
    'alignment': ['alignErr', 'positionErr'],
}

# Physical lane mapping: (bay, lane) -> hardware info
# Source: atca_amc_carrier_gen2_c03.pdf schematic + kintexuplus XDC constraints
# Components in signal path: FPGA GTH pin -> AC-coupling cap -> AMC connector -> AMC board
LANE_HW_MAP = {
    # Bay 0: P11 (lanes 0-5), P12 (lanes 6-9)
    (0, 0): {'fpga_tx': 'AL4/AL3', 'fpga_rx': 'AK2/AK1', 'bank': 224, 'ch': 2,
             'net': 'JESD0',  'rx_caps': 'C1129/C1130', 'conn': 'P11'},
    (0, 1): {'fpga_tx': 'AK6/AK5', 'fpga_rx': 'AJ4/AJ3', 'bank': 224, 'ch': 3,
             'net': 'JESD1',  'rx_caps': 'C1133/C1134', 'conn': 'P11'},
    (0, 2): {'fpga_tx': 'AH6/AH5', 'fpga_rx': 'AH2/AH1', 'bank': 225, 'ch': 0,
             'net': 'JESD2',  'rx_caps': 'C1137/C1138', 'conn': 'P11'},
    (0, 3): {'fpga_tx': 'AG4/AG3', 'fpga_rx': 'AF2/AF1', 'bank': 225, 'ch': 1,
             'net': 'JESD3',  'rx_caps': 'C1141/C1142', 'conn': 'P11'},
    (0, 4): {'fpga_tx': 'AN4/AN3', 'fpga_rx': 'AP2/AP1', 'bank': 224, 'ch': 0,
             'net': 'JESD4',  'rx_caps': 'C1209/C1210', 'conn': 'P11'},
    (0, 5): {'fpga_tx': 'AM6/AM5', 'fpga_rx': 'AM2/AM1', 'bank': 224, 'ch': 1,
             'net': 'JESD5',  'rx_caps': 'C1213/C1214', 'conn': 'P11'},
    (0, 6): {'fpga_tx': 'AE4/AE3', 'fpga_rx': 'AD2/AD1', 'bank': 225, 'ch': 2,
             'net': 'JESD6',  'rx_caps': 'C1217/C1218', 'conn': 'P12'},
    (0, 7): {'fpga_tx': 'AC4/AC3', 'fpga_rx': 'AB2/AB1', 'bank': 225, 'ch': 3,
             'net': 'JESD7',  'rx_caps': 'C1161/C1162', 'conn': 'P12'},
    (0, 8): {'fpga_tx': 'AA4/AA3', 'fpga_rx': 'Y2/Y1',   'bank': 226, 'ch': 0,
             'net': 'JESD8',  'rx_caps': 'C1165/C1166', 'conn': 'P12'},
    (0, 9): {'fpga_tx': 'W4/W3',   'fpga_rx': 'V2/V1',   'bank': 226, 'ch': 1,
             'net': 'JESD9',  'rx_caps': 'C1169/C1170', 'conn': 'P12'},
    # Bay 1: P13 (lanes 0-5), P14 (lanes 6-9)
    (1, 0): {'fpga_tx': 'N4/N3',   'fpga_rx': 'M2/M1',   'bank': 227, 'ch': 0,
             'net': 'JESD10', 'rx_caps': 'C1193/C1194', 'conn': 'P13'},
    (1, 1): {'fpga_tx': 'L4/L3',   'fpga_rx': 'K2/K1',   'bank': 227, 'ch': 1,
             'net': 'JESD11', 'rx_caps': 'C1197/C1198', 'conn': 'P13'},
    (1, 2): {'fpga_tx': 'J4/J3',   'fpga_rx': 'H2/H1',   'bank': 227, 'ch': 2,
             'net': 'JESD12', 'rx_caps': 'C1201/C1202', 'conn': 'P13'},
    (1, 3): {'fpga_tx': 'G4/G3',   'fpga_rx': 'F2/F1',   'bank': 227, 'ch': 3,
             'net': 'JESD13', 'rx_caps': 'C1205/C1206', 'conn': 'P13'},
    (1, 4): {'fpga_tx': 'U4/U3',   'fpga_rx': 'T2/T1',   'bank': 226, 'ch': 2,
             'net': 'JESD14', 'rx_caps': 'C1177/C1178', 'conn': 'P13'},
    (1, 5): {'fpga_tx': 'R4/R3',   'fpga_rx': 'P2/P1',   'bank': 226, 'ch': 3,
             'net': 'JESD15', 'rx_caps': 'C1181/C1182', 'conn': 'P13'},
    (1, 6): {'fpga_tx': 'F6/F5',   'fpga_rx': 'E4/E3',   'bank': 228, 'ch': 0,
             'net': 'JESD16', 'rx_caps': 'C1185/C1186', 'conn': 'P14'},
    (1, 7): {'fpga_tx': 'D6/D5',   'fpga_rx': 'D2/D1',   'bank': 228, 'ch': 1,
             'net': 'JESD17', 'rx_caps': 'C1225/C1226', 'conn': 'P14'},
    (1, 8): {'fpga_tx': 'C4/C3',   'fpga_rx': 'B2/B1',   'bank': 228, 'ch': 2,
             'net': 'JESD18', 'rx_caps': 'C1229/C1230', 'conn': 'P14'},
    (1, 9): {'fpga_tx': 'B6/B5',   'fpga_rx': 'A4/A3',   'bank': 228, 'ch': 3,
             'net': 'JESD19', 'rx_caps': 'C1233/C1234', 'conn': 'P14'},
}

# Default register values from defaults_c03_lb_lb.yml for comparison
DEFAULTS = {
    'JesdRx': {
        'Enable': 0x3F3,
        'Polarity': 0x100,
        'InvertAdcData': 0x00,
        'LinkErrMask': 0x3F,
        'ScrambleEnable': 0x1,
        'InvertSync': 1,  # "Inverted"
        'ReplaceEnable': 1,  # "Enabled"
        'SubClass': 0x1,
        'SysrefDelay': 0x8,
    },
    'JesdTx': {
        'Enable': 0x3CF,
        'Polarity': 0x82,
        'ScrambleEnable': 0x1,
        'InvertSync': 0x0,
        'ReplaceEnable': 0x1,
        'SubClass': 0x1,
        'txDiffCtrl': 0xFF,
        'txPreCursor': 0x05,
        'txPostCursor': 0x05,
    },
    'ADC': {
        'JESD_OUTPUT_SWING': 0x4,
        'SYSREF_DEL_LO': 0x5,
        'SEL_EMP_LANE0': 0x5,
        'SEL_EMP_LANE2': 0x5,
    },
    'DAC': {
        'DacReg[61]': 0x00AD,  # Fully adaptive EQ, boost on
    },
}

# FSM state transitions that can be inferred (from JesdSyncFsmRx.vhd)
# DATA_S -> IDLE_S triggers:
#   nSyncAny_i=0, linkErr=1, enable=0, kStable=1, gtReady=0
# Each of these represents a different failure mode.


class JesdDiagTune:
    """
    Comprehensive JESD204B link diagnostic and adaptive tuning for SMuRF C03.

    The script operates in phases:
      1. SYSTEM SNAPSHOT: Read all JESD-relevant registers and verify config
      2. BASELINE: Measure resync rates on all lanes over extended dwell
      3. ERROR CLASSIFICATION: Read per-lane sticky error bits
      4. SYSREF QUALITY: Verify SYSREF periodicity
      5. ELASTIC BUFFER ANALYSIS: Map latency distribution
      6. LINK ERROR ISOLATION: Mask/unmask individual error types
      7. PER-LANE ISOLATION: Identify the weakest lane(s)
      8. TX LINK HEALTH: Check FPGA->DAC direction
      9. ADAPTIVE SWEEP: Based on findings, sweep relevant parameters
     10. COMBINATORIAL OPTIMIZATION: Multi-parameter joint sweep
     11. FINAL VALIDATION: Extended measurement at best settings
     12. REPORTING: Full diagnostic report + plots
    """

    def __init__(self, S, bays=None, output_dir=None, diag_dwell=60,
                 baseline_dwell=600, sweep_dwell=120,
                 extended_validation_dwell=300, verbose=True):
        """
        Args:
            S: pysmurf SmurfControl instance.
            bays: List of AMC bays to diagnose, e.g. [0, 1]. Default: [0, 1].
            output_dir: Output directory. Default: S.output_dir/jesd_tuning_{timestamp}/
            diag_dwell: Seconds per diagnostic measurement (phases 3-8).
            baseline_dwell: Seconds for initial baseline rate measurement
                (default 600 = 10 min, catches rare events).
            sweep_dwell: Seconds per sweep point. 120s recommended for
                events occurring ~1/min.
            extended_validation_dwell: Seconds for final validation. 300s
                recommended.
            verbose: Print progress to stdout.
        """
        self.S = S
        if bays is None:
            bays = [0, 1]
        self.bays = bays
        self.verbose = verbose
        self.diag_dwell = diag_dwell
        self.baseline_dwell = baseline_dwell
        self.sweep_dwell = sweep_dwell
        self.validation_dwell = extended_validation_dwell

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_dir is None:
            try:
                base = Path(S.output_dir)
            except (AttributeError, TypeError):
                base = Path('.')
            output_dir = base / f'jesd_tuning_{ts}'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Incremental report file (crash-safe: flushed after each phase)
        self._report_path = self.output_dir / 'progress_report.txt'
        self._report_file = open(self._report_path, 'w')
        self._report(f'JESD204B Diagnostic & Tuning — Started {datetime.now().isoformat()}')
        self._report(f'Bays: {bays}, Baseline dwell: {baseline_dwell}s, '
                     f'Diag dwell: {diag_dwell}s, Sweep dwell: {sweep_dwell}s, '
                     f'Validation: {extended_validation_dwell}s')
        self._report('=' * 70)

        # Per-bay PV roots will be set by _set_bay() before each phase
        self.bay = bays[0]
        self._epics = S.epics_root + ':'
        self._set_bay(self.bay)

        # Print system identification
        self._print_system_id()

    def _set_bay(self, bay):
        """Switch internal PV roots to target a specific bay."""
        self.bay = bay
        self._rx_root = self.S.jesd_rx_root.format(bay)
        self._tx_root = self.S.jesd_tx_root.format(bay)
        self._lmk_root = self.S.lmk.format(bay)
        self._mmcore = self.S.microwave_mux_core.format(bay)

    def _print_system_id(self):
        """Print carrier/AMC identification at startup."""
        self._report('')
        self._report('SYSTEM IDENTIFICATION')
        self._report('-' * 40)

        id_lines = []
        id_lines.append(f'Server:        {self.S.epics_root}')

        # Carrier SN (via ATCA monitor or shell)
        try:
            carrier_sn = self.S.get_carrier_sn(use_shell=True)
            id_lines.append(f'Carrier SN:    {carrier_sn}')
        except Exception:
            pass

        # Slot number
        try:
            id_lines.append(f'Slot:          {self.S.slot_number}')
        except Exception:
            pass

        # Per-bay AMC info
        for bay in self.bays:
            parts = []
            try:
                amc_sn = self.S.get_amc_sn(bay=bay, use_shell=True)
                parts.append(f'SN={amc_sn}')
            except Exception:
                pass
            try:
                rx_en = int(self.S._caget(
                    self.S.jesd_rx_root.format(bay) + 'Enable'))
                tx_en = int(self.S._caget(
                    self.S.jesd_tx_root.format(bay) + 'Enable'))
                parts.append(f'RX=0x{rx_en:03X}')
                parts.append(f'TX=0x{tx_en:03X}')
            except Exception:
                pass
            id_lines.append(f'Bay {bay}:          {", ".join(parts)}')

        for line in id_lines:
            self._log(line)
            self._report(f'  {line}')
        self._report('')

    # =========================================================================
    # Low-level register access
    # =========================================================================

    def _get(self, pv_suffix):
        return self.S._caget(self._epics + pv_suffix)

    def _set(self, pv_suffix, val):
        self.S._caput(self._epics + pv_suffix, val)

    def _rx_get(self, reg):
        return self.S._caget(self._rx_root + reg)

    def _rx_set(self, reg, val):
        self.S._caput(self._rx_root + reg, val)

    def _tx_get(self, reg):
        return self.S._caget(self._tx_root + reg)

    def _tx_set(self, reg, val):
        self.S._caput(self._tx_root + reg, val)

    def _adc_get(self, adc, reg):
        path = self._mmcore + f'ADC[{adc}]:' + reg
        return self.S._caget(path)

    def _adc_set(self, adc, reg, val):
        path = self._mmcore + f'ADC[{adc}]:' + reg
        self.S._caput(path, val)

    def _adc_ch_get(self, adc, ch, reg):
        path = self._mmcore + f'ADC[{adc}]:CH[{ch}]:' + reg
        return self.S._caget(path)

    def _adc_ch_set(self, adc, ch, reg, val):
        path = self._mmcore + f'ADC[{adc}]:CH[{ch}]:' + reg
        self.S._caput(path, val)

    def _adc_enable(self, adc, enable=True):
        """Enable/disable ADC rogue device node (required for register access)."""
        path = self._mmcore + f'ADC[{adc}]:enable'
        self.S._caput(path, enable)

    def _adc_enable_all(self):
        """Enable all ADC device nodes for register access."""
        for adc in [0, 1]:
            self._adc_enable(adc, True)

    def _adc_disable_all(self):
        """Disable ADC device nodes (restore default state)."""
        for adc in [0, 1]:
            self._adc_enable(adc, False)

    def _dac_enable(self, dac, enable=True):
        """Enable/disable DAC rogue device node."""
        path = self._mmcore + f'DAC[{dac}]:enable'
        self.S._caput(path, enable)

    def _dac_enable_all(self):
        for dac in [0, 1]:
            self._dac_enable(dac, True)

    def _dac_disable_all(self):
        for dac in [0, 1]:
            self._dac_enable(dac, False)

    def _lmk_enable(self, enable=True):
        """Enable/disable LMK rogue device node."""
        path = self._mmcore + 'LMK:enable'
        self.S._caput(path, enable)

    def _dac_get(self, dac, reg):
        path = self._mmcore + f'DAC[{dac}]:' + reg
        return self.S._caget(path)

    def _dac_set(self, dac, reg, val):
        path = self._mmcore + f'DAC[{dac}]:' + reg
        self.S._caput(path, val)

    def _lmk_get(self, reg):
        return self.S._caget(self._lmk_root + reg)

    def _lmk_set(self, reg, val):
        self.S._caput(self._lmk_root + reg, val)

    def _log(self, msg, level=0):
        if self.verbose:
            indent = '  ' * level
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'[{ts}] {indent}{msg}')

    def _wait(self, seconds, label='Measuring'):
        """Sleep with a live countdown bar on the terminal."""
        if not self.verbose or seconds < 2:
            time.sleep(seconds)
            return
        width = 30
        start = time.time()
        try:
            while True:
                elapsed = time.time() - start
                remaining = seconds - elapsed
                if remaining <= 0:
                    break
                frac = elapsed / seconds
                filled = int(frac * width)
                bar = '█' * filled + '░' * (width - filled)
                mins, secs = divmod(int(remaining), 60)
                ts = datetime.now().strftime('%H:%M:%S')
                sys.stdout.write(
                    f'\r[{ts}]   {label} |{bar}| '
                    f'{mins:02d}:{secs:02d} remaining   ')
                sys.stdout.flush()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()
            raise
        # Clear the line when done
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()

    def _report(self, msg):
        """Write to the incremental progress report (flush-on-write for crash safety)."""
        self._report_file.write(msg + '\n')
        self._report_file.flush()

    def _report_phase(self, phase_name, data):
        """Write a phase summary to the progress report."""
        self._report(f'\n{"=" * 70}')
        self._report(f'{phase_name} — {datetime.now().strftime("%H:%M:%S")}')
        self._report('=' * 70)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and len(str(v)) > 200:
                    self._report(f'  {k}:')
                    for k2, v2 in v.items():
                        self._report(f'    {k2}: {v2}')
                else:
                    self._report(f'  {k}: {v}')
        else:
            self._report(f'  {data}')
        self._report_file.flush()

    # =========================================================================
    # Link management
    # =========================================================================

    def _verify_link_up(self, enable_mask=None, max_retries=3):
        """
        Verify JESD RX link is healthy after a configuration change.
        Checks: DataValid == Enable (all enabled lanes have valid data).
        Returns True if link is up, False if not (after retries).
        """
        if enable_mask is None:
            enable_mask = DEFAULT_RX_ENABLE
        for attempt in range(max_retries):
            time.sleep(0.3)
            data_valid = int(self._rx_get('DataValid'))
            # DataValid should match the enabled lane mask
            if (data_valid & enable_mask) == enable_mask:
                return True
            if attempt < max_retries - 1:
                # Try recovering
                self._recover_link()
        return False

    def _verify_tx_link_up(self, enable_mask=None, max_retries=3):
        """Verify JESD TX link is healthy."""
        if enable_mask is None:
            enable_mask = DEFAULT_TX_ENABLE
        for attempt in range(max_retries):
            time.sleep(0.3)
            data_valid = int(self._tx_get('DataValid'))
            if (data_valid & enable_mask) == enable_mask:
                return True
            if attempt < max_retries - 1:
                self._recover_tx_link()
        return False

    def _clear_errors(self):
        """Toggle ClearErrors to reset all sticky bits and StatusValidCnt."""
        self._rx_set('ClearErrors', 1)
        time.sleep(0.05)
        self._rx_set('ClearErrors', 0)
        time.sleep(0.1)

    def _clear_tx_errors(self):
        """Toggle TX ClearErrors."""
        self._tx_set('ClearErrors', 1)
        time.sleep(0.05)
        self._tx_set('ClearErrors', 0)
        time.sleep(0.1)

    def _recover_link(self, settle_time=1.0, enable_mask=None):
        """
        Recover JESD RX link with proper settling for GT CDR lock.

        With continuous SYSREF, the FSM will re-sync on the next natural
        SYSREF edge once all preconditions are met (gtReady, kStable).
        The critical timing is:
          1. Enable toggle resets GT → GT needs ~100ms to re-lock PLLs
          2. nSync goes low → ADC detects SYNC~ assertion, starts sending K28.5
          3. CDR locks onto K stream → s_kStable after 4 consecutive K detects
          4. Next SYSREF edge → FSM transitions IDLE->SYSREF->SYNC->...->DATA

        The explicit PwrUpSysRef is technically redundant with continuous SYSREF
        but included for robustness (re-aligns LMK SYSREF divider if it drifted).
        """
        if enable_mask is None:
            enable_mask = DEFAULT_RX_ENABLE
        # Disable RX (asserts nSync low to ADCs)
        self._rx_set('Enable', 0x0)
        time.sleep(0.1)
        # Re-enable with requested mask
        self._rx_set('Enable', enable_mask)
        # Wait for GT rstDone + CDR lock + K28.5 detection (4 cycles @ ~6Gbps)
        time.sleep(0.3)
        # Fire SYSREF for robustness (re-arms LMFC alignment)
        # Note: PwrUpSysRef is a command-type PV that works without LMK enable
        self.S.run_pwr_up_sys_ref(self.bay)
        # Wait for FSM to traverse CGS->ILAS(4 multiframes)->DATA
        time.sleep(settle_time)

    def _recover_tx_link(self):
        """Recover TX (FPGA->DAC) link."""
        self._tx_set('Enable', 0x0)
        time.sleep(0.1)
        self._tx_set('Enable', DEFAULT_TX_ENABLE)
        time.sleep(0.1)
        # Toggle DAC JESD reset (need DAC nodes enabled)
        self._dac_enable_all()
        for dac in [0, 1]:
            self._dac_set(dac, 'JesdRstN', 0x0)
            time.sleep(0.05)
            self._dac_set(dac, 'JesdRstN', 0x1)
        self._dac_disable_all()
        time.sleep(0.1)
        self.S.run_pwr_up_sys_ref(self.bay)
        time.sleep(0.5)

    # =========================================================================
    # Measurement primitives
    # =========================================================================

    def _read_rx_valid_cnts(self):
        """Read StatusValidCnt for all enabled RX lanes."""
        cnts = {}
        for lane in ENABLED_RX_LANES:
            cnts[lane] = int(self.S.get_jesd_rx_status_valid_cnt(self.bay, lane))
        return cnts

    def _read_tx_valid_cnts(self):
        """Read StatusValidCnt for all enabled TX lanes."""
        cnts = {}
        for lane in ENABLED_TX_LANES:
            cnts[lane] = int(self.S.get_jesd_tx_status_valid_cnt(self.bay, lane))
        return cnts

    def _measure_rx_rate(self, dwell=None, label='RX measurement',
                         enable_mask=None):
        """
        Clear errors, verify link is up, wait `dwell` seconds, return
        per-lane resync rate (StatusValidCnt increments per second).
        If link cannot be established, returns infinity for all lanes.
        """
        if dwell is None:
            dwell = self.diag_dwell
        if enable_mask is None:
            enable_mask = int(self._rx_get('Enable'))
        self._clear_errors()
        time.sleep(0.3)
        # Verify link is actually up before measuring
        if not self._verify_link_up(enable_mask):
            self._log('  WARNING: Link not up, measurement invalid', level=2)
            return {lane: float('inf') for lane in ENABLED_RX_LANES}
        t0 = time.time()
        cnts0 = self._read_rx_valid_cnts()
        self._wait(dwell, label)
        t1 = time.time()
        cnts1 = self._read_rx_valid_cnts()
        dt = t1 - t0
        rates = {}
        for lane in ENABLED_RX_LANES:
            rates[lane] = (cnts1[lane] - cnts0[lane]) / dt
        return rates

    def _measure_tx_rate(self, dwell=None, label='TX measurement'):
        """Measure TX resync rate."""
        if dwell is None:
            dwell = self.diag_dwell
        self._clear_tx_errors()
        time.sleep(0.5)
        t0 = time.time()
        cnts0 = self._read_tx_valid_cnts()
        self._wait(dwell, label)
        t1 = time.time()
        cnts1 = self._read_tx_valid_cnts()
        dt = t1 - t0
        rates = {}
        for lane in ENABLED_TX_LANES:
            rates[lane] = (cnts1[lane] - cnts0[lane]) / dt
        return rates

    def _total_rate(self, rates):
        return sum(rates.values())

    def _best_index(self, rates_data, values, original):
        """
        Find the best (lowest-rate) index from sweep data.
        If all rates are equal (e.g. all zero), prefer the original value.
        """
        min_rate = min(rates_data)
        # If all rates are the same, no preference — keep original
        if min(rates_data) == max(rates_data):
            if original in values:
                return values.index(original)
            # Original not in sweep range; pick middle
            return len(values) // 2
        return int(np.argmin(rates_data))

    def _read_sysref_period(self):
        """Read SysRefPeriod min/max from RX module."""
        pmin = int(self._rx_get('SysRefPeriodmin'))
        pmax = int(self._rx_get('SysRefPeriodmax'))
        return pmin, pmax

    def _read_tx_sysref_period(self):
        """Read SysRefPeriod min/max from TX module."""
        pmin = int(self._tx_get('SysRefPeriodmin'))
        pmax = int(self._tx_get('SysRefPeriodmax'))
        return pmin, pmax

    def _read_per_lane_status(self):
        """
        Read per-lane status registers from the JesdRx module.

        Some registers are bitmasks (one bit per lane, read once):
            GTReady, DataValid, AlignErr, nSync, RxBuffUfl, RxBuffOfl,
            PositionErr, RxEnabled, SysRefDetected, CommaDetected
        Others are per-lane indexed arrays:
            DisparityErr[n], NotInTableErr[n], ElBuffLatency[n]
        """
        # Read bitmask registers once
        gt_ready = int(self._rx_get('GTReady'))
        data_valid = int(self._rx_get('DataValid'))
        align_err = int(self._rx_get('AlignErr'))
        nsync = int(self._rx_get('nSync'))
        buf_ufl = int(self._rx_get('RxBuffUfl'))
        buf_ofl = int(self._rx_get('RxBuffOfl'))
        pos_err = int(self._rx_get('PositionErr'))
        rx_enabled = int(self._rx_get('RxEnabled'))
        sysref_det = int(self._rx_get('SysRefDetected'))
        comma_det = int(self._rx_get('CommaDetected'))

        status = {}
        for lane in ENABLED_RX_LANES:
            s = {}
            # Extract per-lane bit from bitmasks
            s['GTReady'] = (gt_ready >> lane) & 1
            s['DataValid'] = (data_valid >> lane) & 1
            s['AlignErr'] = (align_err >> lane) & 1
            s['nSync'] = (nsync >> lane) & 1
            s['RxBuffUfl'] = (buf_ufl >> lane) & 1
            s['RxBuffOfl'] = (buf_ofl >> lane) & 1
            s['PositionErr'] = (pos_err >> lane) & 1
            s['RxEnabled'] = (rx_enabled >> lane) & 1
            s['SysRefDetected'] = (sysref_det >> lane) & 1
            s['CommaDetected'] = (comma_det >> lane) & 1
            # Per-lane indexed registers
            s['DisparityErr'] = int(self._rx_get(f'DisparityErr[{lane}]'))
            s['NotInTableErr'] = int(self._rx_get(f'NotInTableErr[{lane}]'))
            s['ElBuffLatency'] = int(self._rx_get(f'ElBuffLatency[{lane}]'))
            status[lane] = s
        return status

    def _read_tx_per_lane_status(self):
        """Read per-lane TX status (bitmask registers, extracted per lane)."""
        gt_ready = int(self._tx_get('GTReady'))
        data_valid = int(self._tx_get('DataValid'))
        nsync = int(self._tx_get('nSync'))
        tx_enabled = int(self._tx_get('TxEnabled'))
        sysref_det = int(self._tx_get('SysRefDetected'))

        status = {}
        for lane in ENABLED_TX_LANES:
            s = {}
            s['GTReady'] = (gt_ready >> lane) & 1
            s['DataValid'] = (data_valid >> lane) & 1
            s['nSync'] = (nsync >> lane) & 1
            s['TxEnabled'] = (tx_enabled >> lane) & 1
            s['SysRefDetected'] = (sysref_det >> lane) & 1
            status[lane] = s
        return status

    # =========================================================================
    # Phase 1: System Snapshot
    # =========================================================================

    def _snapshot_system(self):
        """Read all JESD-relevant registers and verify against expected defaults."""
        self._log('=' * 70)
        self._log('PHASE 1: SYSTEM CONFIGURATION SNAPSHOT')
        self._log('=' * 70)

        snap = {'rx': {}, 'tx': {}, 'adc': {}, 'dac': {}, 'lmk': {},
                'mismatches': []}

        # JesdRx registers
        rx_regs = ['Enable', 'SysrefDelay', 'SubClass', 'ReplaceEnable',
                   'ResetGTs', 'ClearErrors', 'InvertSync', 'ScrambleEnable',
                   'LinkErrMask', 'InvertAdcData']
        for reg in rx_regs:
            try:
                snap['rx'][reg] = int(self._rx_get(reg))
            except Exception as e:
                snap['rx'][reg] = f'ERROR: {e}'

        # JesdTx registers
        tx_regs = ['Enable', 'SubClass', 'ReplaceEnable', 'ScrambleEnable',
                   'InvertSync', 'InvertDacData']
        for reg in tx_regs:
            try:
                snap['tx'][reg] = int(self._tx_get(reg))
            except Exception as e:
                snap['tx'][reg] = f'ERROR: {e}'

        # TX emphasis per lane (lowercase in rogue device tree)
        for lane in ENABLED_TX_LANES:
            snap['tx'][f'txDiffCtrl[{lane}]'] = int(self._tx_get(f'txDiffCtrl[{lane}]'))
            snap['tx'][f'txPreCursor[{lane}]'] = int(self._tx_get(f'txPreCursor[{lane}]'))
            snap['tx'][f'txPostCursor[{lane}]'] = int(self._tx_get(f'txPostCursor[{lane}]'))

        # ADC settings (both ADCs) — must enable rogue nodes first
        self._adc_enable_all()
        for adc in [0, 1]:
            prefix = f'ADC[{adc}]'
            snap['adc'][f'{prefix}:JESD_OUTPUT_SWING'] = int(self._adc_get(adc, 'JESD_OUTPUT_SWING'))
            snap['adc'][f'{prefix}:SYSREF_DEL_EN'] = int(self._adc_get(adc, 'SYSREF_DEL_EN'))
            snap['adc'][f'{prefix}:SYSREF_DEL_HI'] = int(self._adc_get(adc, 'SYSREF_DEL_HI'))
            snap['adc'][f'{prefix}:SYSREF_DEL_LO'] = int(self._adc_get(adc, 'SYSREF_DEL_LO'))
            snap['adc'][f'{prefix}:SYNCB_POL'] = int(self._adc_get(adc, 'SYNCB_POL'))
            for ch in [0, 1]:
                snap['adc'][f'{prefix}:CH[{ch}]:SCRAMBLE_EN'] = int(
                    self._adc_ch_get(adc, ch, 'SCRAMBLE_EN'))
                snap['adc'][f'{prefix}:CH[{ch}]:SEL_EMP_LANE0'] = int(
                    self._adc_ch_get(adc, ch, 'SEL_EMP_LANE0'))
                snap['adc'][f'{prefix}:CH[{ch}]:SEL_EMP_LANE2'] = int(
                    self._adc_ch_get(adc, ch, 'SEL_EMP_LANE2'))
        self._adc_disable_all()

        # DAC JESD-relevant registers
        self._dac_enable_all()
        dac_jesd_regs = [4, 5, 36, 37, 59, 60, 61, 62, 63, 74, 75, 76, 77, 78,
                         81, 82, 84, 85, 87, 88, 90, 91, 92, 95, 96]
        for dac in [0, 1]:
            for reg_num in dac_jesd_regs:
                key = f'DAC[{dac}]:DacReg[{reg_num}]'
                snap['dac'][key] = int(self._dac_get(dac, f'DacReg[{reg_num}]'))
        self._dac_disable_all()

        # LMK registers
        self._lmk_enable(True)
        snap['lmk']['EnableSync'] = int(self._lmk_get('EnableSync'))
        snap['lmk']['EnableSysRef'] = int(self._lmk_get('EnableSysRef'))
        self._lmk_enable(False)

        # Check critical consistency
        self._log('Checking configuration consistency...')
        checks = []

        # Scramble must match ADC<->FPGA
        if snap['rx'].get('ScrambleEnable') != 1:
            checks.append('CRITICAL: RX ScrambleEnable != 1')
        if snap['tx'].get('ScrambleEnable') != 1:
            checks.append('CRITICAL: TX ScrambleEnable != 1')

        # SubClass must be consistent
        if snap['rx'].get('SubClass') != snap['tx'].get('SubClass'):
            checks.append('CRITICAL: RX/TX SubClass mismatch')

        # Enable mask sanity
        if snap['rx'].get('Enable') != DEFAULT_RX_ENABLE:
            checks.append(f'NOTE: RX Enable=0x{snap["rx"].get("Enable", 0):X} '
                          f'differs from default 0x{DEFAULT_RX_ENABLE:X}')
        if snap['tx'].get('Enable') != DEFAULT_TX_ENABLE:
            checks.append(f'NOTE: TX Enable=0x{snap["tx"].get("Enable", 0):X} '
                          f'differs from default 0x{DEFAULT_TX_ENABLE:X}')

        # DAC EQ settings
        for dac in [0, 1]:
            eq = snap['dac'].get(f'DAC[{dac}]:DacReg[61]', 0)
            if eq != 0x00AD:
                checks.append(f'NOTE: DAC[{dac}] EQ reg[61]=0x{eq:04X} '
                              f'(expected 0x00AD fully adaptive)')

        snap['mismatches'] = checks
        for c in checks:
            self._log(f'  {c}', level=1)

        if not checks:
            self._log('  All configuration checks passed.')

        return snap

    # =========================================================================
    # Phase 2: Baseline Measurement
    # =========================================================================

    def _baseline_measurement(self):
        """Extended baseline measurement of resync rates (10 min default)."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 2: BASELINE RESYNC RATE MEASUREMENT')
        self._log('=' * 70)
        self._log(f'Dwell time: {self.baseline_dwell}s ({self.baseline_dwell/60:.0f} min)')

        rates = self._measure_rx_rate(self.baseline_dwell, label='Baseline')
        bad_lanes = [l for l, r in rates.items() if r > 0]

        self._log('Per-lane RX resync rates (increments/sec):')
        for lane in ENABLED_RX_LANES:
            adc_info = ADC_LANE_MAP.get(lane, {}).get('desc', '?')
            marker = ' <<< ACTIVE' if rates[lane] > 0 else ''
            self._log(f'  Lane {lane:2d} ({adc_info}): '
                      f'{rates[lane]:10.6f} inc/s{marker}', level=1)

        self._log(f'\nTotal RX rate: {self._total_rate(rates):.6f} inc/s')
        self._log(f'Bad lanes: {bad_lanes}')

        return {'rates': rates, 'bad_lanes': bad_lanes,
                'total_rate': self._total_rate(rates)}

    # =========================================================================
    # Phase 3: Error Classification
    # =========================================================================

    def _classify_errors(self):
        """Read per-lane sticky error bits after allowing errors to accumulate."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 3: PER-LANE ERROR CLASSIFICATION')
        self._log('=' * 70)

        # Clear, wait, then read sticky bits
        self._clear_errors()
        self._log(f'Waiting {self.diag_dwell}s for errors to accumulate...')
        self._wait(self.diag_dwell, 'Error accumulation')

        status = self._read_per_lane_status()

        self._log('Per-lane error flags:')
        err_summary = defaultdict(list)
        for lane in ENABLED_RX_LANES:
            s = status[lane]
            errs = []
            if s['DisparityErr']:
                errs.append(f"dispErr(0x{s['DisparityErr']:X})")
                err_summary['dispErr'].append(lane)
            if s['NotInTableErr']:
                errs.append(f"decErr(0x{s['NotInTableErr']:X})")
                err_summary['decErr'].append(lane)
            if s['AlignErr']:
                errs.append('alignErr')
                err_summary['alignErr'].append(lane)
            if s['RxBuffUfl']:
                errs.append('bufUnf')
                err_summary['bufUnf'].append(lane)
            if s['RxBuffOfl']:
                errs.append('bufOvf')
                err_summary['bufOvf'].append(lane)
            if s['PositionErr']:
                errs.append('posErr')
                err_summary['positionErr'].append(lane)

            err_str = ', '.join(errs) if errs else 'clean'
            gt_ok = 'GT_RDY' if s['GTReady'] else 'GT_DOWN'
            dv = 'DV' if s['DataValid'] else 'no_DV'
            self._log(f'  Lane {lane:2d}: [{gt_ok} {dv}] {err_str}', level=1)

        self._log('\nError type summary:')
        for etype, lanes in err_summary.items():
            self._log(f'  {etype}: lanes {lanes}', level=1)

        return {'lane_status': status, 'error_types': dict(err_summary)}

    # =========================================================================
    # Phase 4: SYSREF Quality
    # =========================================================================

    def _check_sysref(self):
        """Check SYSREF periodicity on both RX and TX."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 4: SYSREF QUALITY CHECK')
        self._log('=' * 70)

        # Clear to reset the sysref monitor, wait for fresh stats
        self._clear_errors()
        time.sleep(5)

        rx_min, rx_max = self._read_sysref_period()
        tx_min, tx_max = self._read_tx_sysref_period()

        rx_ok = (rx_min == rx_max) and (rx_min > 0)
        tx_ok = (tx_min == tx_max) and (tx_min > 0)

        self._log(f'  RX SYSREF: min={rx_min}, max={rx_max} '
                  f'[{"OK" if rx_ok else "JITTERY/MISSING"}]')
        self._log(f'  TX SYSREF: min={tx_min}, max={tx_max} '
                  f'[{"OK" if tx_ok else "JITTERY/MISSING"}]')

        if not rx_ok:
            self._log('  WARNING: SYSREF jitter detected. This can cause:')
            self._log('    - Buffer overflow/underflow (timing misalignment)')
            self._log('    - Non-deterministic link establishment')
            self._log('    - Investigate LMK configuration and clock distribution')

        return {
            'rx_min': rx_min, 'rx_max': rx_max, 'rx_ok': rx_ok,
            'tx_min': tx_min, 'tx_max': tx_max, 'tx_ok': tx_ok,
        }

    # =========================================================================
    # Phase 5: Elastic Buffer Analysis
    # =========================================================================

    def _analyze_elastic_buffer(self):
        """Read and analyze elastic buffer latency distribution."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 5: ELASTIC BUFFER LATENCY ANALYSIS')
        self._log('=' * 70)

        status = self._read_per_lane_status()
        latencies = {lane: status[lane]['ElBuffLatency'] for lane in ENABLED_RX_LANES}

        lat_values = list(latencies.values())
        lat_mean = np.mean(lat_values) if lat_values else 0
        lat_std = np.std(lat_values) if lat_values else 0
        lat_spread = max(lat_values) - min(lat_values) if lat_values else 0

        risks = []
        self._log(f'  Mean latency: {lat_mean:.1f} clks, '
                  f'Std: {lat_std:.1f}, Spread: {lat_spread}')

        for lane in ENABLED_RX_LANES:
            lat = latencies[lane]
            risk = ''
            if lat <= 3:
                risk = 'UNDERFLOW RISK (very low)'
                risks.append(('underflow', lane))
            elif lat <= 8:
                risk = 'LOW (monitor)'
            elif lat >= 250:
                risk = 'OVERFLOW RISK (very high)'
                risks.append(('overflow', lane))
            elif lat >= 200:
                risk = 'HIGH (monitor)'
            self._log(f'  Lane {lane:2d}: {lat:3d} clks {risk}', level=1)

        if lat_spread > 20:
            self._log(f'  WARNING: Large inter-lane latency spread ({lat_spread} clks)')
            self._log('    Possible causes: trace length mismatch, CDR lock delay')

        return {
            'latencies': latencies,
            'mean': lat_mean, 'std': lat_std, 'spread': lat_spread,
            'risks': risks,
        }

    # =========================================================================
    # Phase 6: Link Error Isolation
    # =========================================================================

    def _isolate_link_errors(self):
        """
        Systematically mask/unmask each error type in LinkErrMask to determine
        which specific error(s) are triggering re-syncs.
        """
        self._log('\n' + '=' * 70)
        self._log('PHASE 6: LINK ERROR TYPE ISOLATION')
        self._log('=' * 70)

        original_mask = int(self._rx_get('LinkErrMask'))
        short_dwell = max(20, self.diag_dwell // 3)
        results = {'original_mask': original_mask}

        # Test 1: Mask ALL errors (should stabilize if errors are data-level)
        self._log(f'  Test: All errors masked (0x00), dwell={short_dwell}s')
        self._rx_set('LinkErrMask', 0x00)
        self._recover_link()
        time.sleep(2)
        rates_all_masked = self._measure_rx_rate(short_dwell, label='All masked')
        total_masked = self._total_rate(rates_all_masked)
        results['all_masked_rate'] = total_masked
        self._log(f'    Result: {total_masked:.6f} inc/s')

        if total_masked > 0:
            self._log('    *** CRITICAL: Link drops with ALL errors masked! ***')
            self._log('    This means the FSM is leaving DATA_S due to:')
            self._log('      - GT not ready (PLL unlock / CDR loss)')
            self._log('      - s_kStable=1 (false K28.5 detection in data)')
            self._log('      - nSyncAny_i=0 (another lane pulling global sync down)')
            results['gt_or_pll_unstable'] = True
            self._rx_set('LinkErrMask', original_mask)
            self._recover_link()
            return results

        results['gt_or_pll_unstable'] = False
        self._log('    Link stable with all errors masked (GT/PLL OK)')

        # Test 2: Enable each error type individually
        self._log(f'\n  Enabling error types one at a time:')
        per_error = {}
        for bit, name in LINK_ERR_BITS.items():
            mask = 1 << bit
            self._rx_set('LinkErrMask', mask)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(short_dwell, label=f'Isolate {name}')
            total = self._total_rate(rates)
            triggers = total > 0
            per_error[name] = {
                'mask': mask,
                'total_rate': total,
                'per_lane': dict(rates),
                'triggers_resync': triggers,
            }
            status_str = f'TRIGGERS ({total:.4f} inc/s)' if triggers else 'stable'
            self._log(f'    {name:12s} (mask=0x{mask:02X}): {status_str}', level=1)

        results['per_error'] = per_error

        # Test 3: Cumulative enable (build up from most to least common)
        self._log('\n  Cumulative error enable test:')
        cumulative_mask = 0x00
        cumulative_results = []
        # Order: dispErr first (most common SI), then decErr, then rest
        err_order = ['dispErr', 'decErr', 'alignErr', 'bufUnf', 'bufOvf', 'positionErr']
        bit_for_name = {v: k for k, v in LINK_ERR_BITS.items()}

        for name in err_order:
            bit = bit_for_name[name]
            cumulative_mask |= (1 << bit)
            self._rx_set('LinkErrMask', cumulative_mask)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(short_dwell, label=f'Cumul +{name}')
            total = self._total_rate(rates)
            cumulative_results.append({
                'added': name, 'mask': cumulative_mask, 'rate': total
            })
            self._log(f'    +{name:12s} (mask=0x{cumulative_mask:02X}): '
                      f'{total:.4f} inc/s', level=1)

        results['cumulative'] = cumulative_results

        # Restore original mask
        self._rx_set('LinkErrMask', original_mask)
        self._recover_link()

        return results

    # =========================================================================
    # Phase 7: Per-Lane Isolation
    # =========================================================================

    def _isolate_lanes(self, bad_lanes):
        """
        Disable bad lanes one at a time to find which are dragging the link
        down via global nSync, then try disabling all significant offenders
        simultaneously to see if a stable subset exists.
        Reports hardware info for problematic lanes.
        """
        self._log('\n' + '=' * 70)
        self._log('PHASE 7: PER-LANE ISOLATION')
        self._log('=' * 70)

        if not bad_lanes:
            self._log('  No bad lanes to isolate.')
            return {}

        short_dwell = max(20, self.diag_dwell // 3)
        results = {}

        # Baseline with all lanes
        self._log(f'  Baseline (all lanes enabled): measuring...')
        rates_all = self._measure_rx_rate(short_dwell, label='Lane iso baseline')
        baseline_rate = self._total_rate(rates_all)
        results['all_enabled'] = baseline_rate
        self._log(f'    Rate: {baseline_rate:.4f} inc/s')

        # --- Step 1: Disable each bad lane one at a time ---
        self._log('\n  Step 1: Single-lane disable...')
        per_lane_improvement = {}
        for target_lane in bad_lanes:
            new_mask = DEFAULT_RX_ENABLE & ~(1 << target_lane)
            self._rx_set('Enable', new_mask)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(short_dwell, label=f'Without lane {target_lane}',
                                          enable_mask=new_mask)
            total = self._total_rate(rates)
            improvement = baseline_rate - total
            results[f'without_lane_{target_lane}'] = {
                'mask': new_mask, 'rate': total, 'improvement': improvement
            }
            per_lane_improvement[target_lane] = improvement
            improved = 'IMPROVED' if total < baseline_rate * 0.5 else 'minimal change'
            self._log(f'    Disable lane {target_lane}: rate={total:.4f} [{improved}]',
                      level=1)

        # Restore
        self._rx_set('Enable', DEFAULT_RX_ENABLE)
        self._recover_link()

        # --- Step 2: Identify statistically significant offenders ---
        # A lane is "significant" if disabling it reduces rate by more than
        # 1/(N_lanes) of baseline (i.e. more than its "fair share")
        n_bad = len(bad_lanes)
        significance_threshold = baseline_rate / (n_bad + 1)
        significant_lanes = [lane for lane, imp in per_lane_improvement.items()
                             if imp > significance_threshold]
        results['significant_lanes'] = significant_lanes
        results['significance_threshold'] = significance_threshold

        self._log(f'\n  Significance threshold: {significance_threshold:.4f} inc/s')
        self._log(f'  Significant offenders: {significant_lanes}')

        # --- Step 3: Disable all significant offenders simultaneously ---
        if significant_lanes and len(significant_lanes) < len(bad_lanes):
            self._log(f'\n  Step 2: Disabling {len(significant_lanes)} significant '
                      f'lanes simultaneously...')
            combined_mask = DEFAULT_RX_ENABLE
            for lane in significant_lanes:
                combined_mask &= ~(1 << lane)
            self._rx_set('Enable', combined_mask)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(short_dwell,
                                          label='Without all offenders',
                                          enable_mask=combined_mask)
            combined_rate = self._total_rate(rates)
            stable = combined_rate == 0
            results['combined_disable'] = {
                'disabled_lanes': significant_lanes,
                'mask': combined_mask,
                'rate': combined_rate,
                'stable': stable,
            }
            status = 'STABLE' if stable else f'rate={combined_rate:.4f}'
            self._log(f'    Mask=0x{combined_mask:03X}: {status}')

            if not stable and len(significant_lanes) < len(bad_lanes) - 1:
                # Try progressively adding more lanes to the disable list
                self._log('    Still not stable, trying more lanes...')
                # Sort remaining by improvement, add next best
                remaining = sorted(
                    [(l, per_lane_improvement[l]) for l in bad_lanes
                     if l not in significant_lanes],
                    key=lambda x: x[1], reverse=True)
                for add_lane, _ in remaining:
                    combined_mask &= ~(1 << add_lane)
                    significant_lanes.append(add_lane)
                    self._rx_set('Enable', combined_mask)
                    self._recover_link()
                    time.sleep(1)
                    rates = self._measure_rx_rate(
                        short_dwell, label=f'+disable lane {add_lane}',
                        enable_mask=combined_mask)
                    combined_rate = self._total_rate(rates)
                    if combined_rate == 0:
                        results['stable_subset'] = {
                            'disabled': list(significant_lanes),
                            'mask': combined_mask,
                        }
                        self._log(f'    STABLE with mask=0x{combined_mask:03X} '
                                  f'(disabled: {significant_lanes})')
                        break
            elif stable:
                results['stable_subset'] = {
                    'disabled': significant_lanes,
                    'mask': combined_mask,
                }

            # Restore
            self._rx_set('Enable', DEFAULT_RX_ENABLE)
            self._recover_link()

        # --- Step 4: Report hardware info for problematic lanes ---
        self._log(f'\n  Hardware info for problematic lanes:')
        results['hw_info'] = {}
        for lane in (significant_lanes if significant_lanes else bad_lanes[:3]):
            hw = LANE_HW_MAP.get((self.bay, lane), {})
            if hw:
                info = (f"Lane {lane}: {hw['net']} | "
                        f"FPGA RX pins {hw['fpga_rx']} | "
                        f"GTH bank {hw['bank']} ch{hw['ch']} | "
                        f"AC caps {hw['rx_caps']} | "
                        f"Connector {hw['conn']}")
                self._log(f'    {info}', level=1)
                results['hw_info'][lane] = hw
            else:
                self._log(f'    Lane {lane}: (no HW map entry for bay {self.bay})',
                          level=1)

        # Identify worst offender
        if per_lane_improvement:
            worst_lane = max(per_lane_improvement, key=per_lane_improvement.get)
            results['worst_lane'] = worst_lane
            self._log(f'\n  Worst offending lane: {worst_lane} '
                      f'(removing it improves rate by '
                      f'{per_lane_improvement[worst_lane]:.4f} inc/s)')

        return results

    # =========================================================================
    # Phase 8: TX Link Health
    # =========================================================================

    def _check_tx_health(self):
        """Check FPGA->DAC link health."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 8: TX (FPGA->DAC) LINK HEALTH')
        self._log('=' * 70)

        tx_status = self._read_tx_per_lane_status()
        tx_rates = self._measure_tx_rate(dwell=max(15, self.diag_dwell // 4), label="TX health check")

        any_issues = False
        for lane in ENABLED_TX_LANES:
            s = tx_status[lane]
            rate = tx_rates[lane]
            gt_ok = 'GT_RDY' if s['GTReady'] else 'GT_DOWN'
            dv = 'DV' if s['DataValid'] else 'no_DV'
            sync = 'SYNC' if s['nSync'] else 'no_SYNC'
            marker = ' <<< ISSUE' if (not s['DataValid'] or rate > 0) else ''
            if marker:
                any_issues = True
            self._log(f'  Lane {lane}: [{gt_ok} {dv} {sync}] '
                      f'rate={rate:.4f} inc/s{marker}', level=1)

        return {
            'status': tx_status, 'rates': tx_rates,
            'has_issues': any_issues,
            'total_rate': self._total_rate(tx_rates),
        }

    # =========================================================================
    # Phase 9: Adaptive Parameter Sweeps
    # =========================================================================

    def _plan_sweeps(self, findings):
        """Based on all diagnostic findings, plan which sweeps to run."""
        plan = []
        cause = findings.get('root_cause_class', 'unknown')
        baseline_clean = not findings.get('baseline', {}).get('bad_lanes')

        # Always sweep SysrefDelay (universal, low-risk)
        plan.append('sysref_delay')

        if baseline_clean or cause in ('signal_integrity', 'unknown', None):
            plan.append('adc_emphasis')
            plan.append('adc_swing')

        if baseline_clean or cause in ('timing_buffer', 'unknown', None):
            plan.append('adc_sysref_delay')

        if not baseline_clean and (cause == 'signal_integrity' or cause == 'unknown'):
            plan.append('adc_emphasis_fine')

        if findings.get('tx_has_issues'):
            plan.append('tx_pre_cursor')   # sweeps txPreCursor
            plan.append('tx_post_cursor')  # sweeps txPostCursor
            plan.append('tx_diff_ctrl')    # sweeps txDiffCtrl

        return plan

    def _run_sweeps(self, plan, findings):
        """Execute planned sweeps."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 9: ADAPTIVE PARAMETER SWEEPS')
        self._log('=' * 70)
        self._log(f'Sweep plan: {plan}')
        self._log(f'Dwell per point: {self.sweep_dwell}s')

        all_results = []

        for sweep_name in plan:
            self._log(f'\n--- Sweep: {sweep_name} ---')
            if sweep_name == 'sysref_delay':
                result = self._sweep_sysref_delay()
            elif sweep_name == 'adc_emphasis':
                result = self._sweep_adc_emphasis(coarse=True)
            elif sweep_name == 'adc_emphasis_fine':
                # Only run if coarse found a region of interest
                coarse = next((r for r in all_results
                               if r['parameter'] == 'ADC_SEL_EMP_coarse'), None)
                if coarse and coarse['best_rate'] < coarse['rates'][0]:
                    result = self._sweep_adc_emphasis_fine(coarse['best_value'])
                else:
                    self._log('  Skipping fine emphasis (coarse showed no benefit)')
                    continue
            elif sweep_name == 'adc_swing':
                result = self._sweep_adc_swing()
            elif sweep_name == 'adc_sysref_delay':
                result = self._sweep_adc_sysref_delay()
            elif sweep_name == 'tx_pre_cursor':
                result = self._sweep_tx_param('txPreCursor')
            elif sweep_name == 'tx_post_cursor':
                result = self._sweep_tx_param('txPostCursor')
            elif sweep_name == 'tx_diff_ctrl':
                result = self._sweep_tx_diff_ctrl()
            else:
                continue
            all_results.append(result)

        return all_results

    def _sweep_sysref_delay(self):
        """Sweep FPGA SysrefDelay 0-48 in steps of 4."""
        values = list(range(0, 52, 4))
        original = int(self._rx_get('SysrefDelay'))
        rates_data = []
        lat_data = []

        for val in values:
            self._rx_set('SysrefDelay', val)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
            total = self._total_rate(rates)
            status = self._read_per_lane_status()
            lats = {l: status[l]['ElBuffLatency'] for l in ENABLED_RX_LANES}
            rates_data.append(total)
            lat_data.append(lats)
            self._log(f'  SysrefDelay={val:3d}: rate={total:.4f} inc/s, '
                      f'ElBuf mean={np.mean(list(lats.values())):.0f}', level=1)

        best_idx = self._best_index(rates_data, values, original)
        self._rx_set('SysrefDelay', original)
        self._recover_link()

        result = {
            'parameter': 'SysrefDelay',
            'values': values, 'rates': rates_data,
            'el_buf_latencies': lat_data,
            'original': original,
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST: SysrefDelay={result["best_value"]} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    def _sweep_adc_emphasis(self, coarse=True):
        """Sweep ADC output pre-emphasis on active lanes."""
        if coarse:
            values = [0x00, 0x03, 0x05, 0x08, 0x0C, 0x10, 0x18, 0x20, 0x2F, 0x3F]
            param_name = 'ADC_SEL_EMP_coarse'
        else:
            values = list(range(0, 64, 4))
            param_name = 'ADC_SEL_EMP_full'

        # Enable ADC rogue nodes for register access
        self._adc_enable_all()

        # Read originals for both ADCs, both channels
        originals = {}
        for adc in [0, 1]:
            for ch in [0, 1]:
                for reg in ['SEL_EMP_LANE0', 'SEL_EMP_LANE2']:
                    key = (adc, ch, reg)
                    originals[key] = int(self._adc_ch_get(adc, ch, reg))
        original = originals[(0, 0, 'SEL_EMP_LANE0')]

        rates_data = []
        for val in values:
            for adc in [0, 1]:
                for ch in [0, 1]:
                    self._adc_ch_set(adc, ch, 'SEL_EMP_LANE0', val)
                    self._adc_ch_set(adc, ch, 'SEL_EMP_LANE2', val)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
            total = self._total_rate(rates)
            rates_data.append(total)
            self._log(f'  SEL_EMP=0x{val:02X}: rate={total:.4f} inc/s', level=1)

        best_idx = self._best_index(rates_data, values, original)

        # Restore
        for (adc, ch, reg), oval in originals.items():
            self._adc_ch_set(adc, ch, reg, oval)
        self._adc_disable_all()
        self._recover_link()

        result = {
            'parameter': param_name,
            'values': values, 'rates': rates_data,
            'original': originals[(0, 0, 'SEL_EMP_LANE0')],
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST: SEL_EMP=0x{result["best_value"]:02X} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    def _sweep_adc_emphasis_fine(self, center):
        """Fine sweep around the best coarse emphasis value."""
        low = max(0, center - 4)
        high = min(63, center + 5)
        values = list(range(low, high + 1))

        self._adc_enable_all()

        originals = {}
        for adc in [0, 1]:
            for ch in [0, 1]:
                for reg in ['SEL_EMP_LANE0', 'SEL_EMP_LANE2']:
                    originals[(adc, ch, reg)] = int(self._adc_ch_get(adc, ch, reg))
        original = originals[(0, 0, 'SEL_EMP_LANE0')]

        rates_data = []
        for val in values:
            for adc in [0, 1]:
                for ch in [0, 1]:
                    self._adc_ch_set(adc, ch, 'SEL_EMP_LANE0', val)
                    self._adc_ch_set(adc, ch, 'SEL_EMP_LANE2', val)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
            total = self._total_rate(rates)
            rates_data.append(total)
            self._log(f'  SEL_EMP=0x{val:02X}: rate={total:.4f} inc/s', level=1)

        best_idx = self._best_index(rates_data, values, original)

        for (adc, ch, reg), oval in originals.items():
            self._adc_ch_set(adc, ch, reg, oval)
        self._adc_disable_all()
        self._recover_link()

        result = {
            'parameter': 'ADC_SEL_EMP_fine',
            'values': values, 'rates': rates_data,
            'original': originals[(0, 0, 'SEL_EMP_LANE0')],
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST (fine): SEL_EMP=0x{result["best_value"]:02X} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    def _sweep_adc_swing(self):
        """Sweep ADC JESD output swing."""
        # Per AD9680: 0x0=530mV, 0x2=750mV, 0x4=960mV(max), 0x6=905mV
        values = [0x0, 0x2, 0x4, 0x6]
        labels = ['530mV', '750mV', '960mV_max', '905mV']

        self._adc_enable_all()

        originals = {}
        for adc in [0, 1]:
            originals[adc] = int(self._adc_get(adc, 'JESD_OUTPUT_SWING'))
        original = originals[0]

        rates_data = []
        for val in values:
            for adc in [0, 1]:
                self._adc_set(adc, 'JESD_OUTPUT_SWING', val)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
            total = self._total_rate(rates)
            rates_data.append(total)
            idx = values.index(val)
            self._log(f'  SWING=0x{val:X} ({labels[idx]}): rate={total:.4f}', level=1)

        best_idx = self._best_index(rates_data, values, original)

        for adc, oval in originals.items():
            self._adc_set(adc, 'JESD_OUTPUT_SWING', oval)
        self._adc_disable_all()
        self._recover_link()

        result = {
            'parameter': 'JESD_OUTPUT_SWING',
            'values': values, 'rates': rates_data, 'labels': labels,
            'original': originals[0],
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST: SWING=0x{result["best_value"]:X} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    def _sweep_adc_sysref_delay(self):
        """Sweep ADC-side SYSREF fine delay (0-7). Register is UInt3."""
        values = list(range(0, 8))

        self._adc_enable_all()

        originals = {}
        for adc in [0, 1]:
            originals[adc] = int(self._adc_get(adc, 'SYSREF_DEL_LO'))
        original = originals[0]

        rates_data = []
        for val in values:
            for adc in [0, 1]:
                self._adc_set(adc, 'SYSREF_DEL_LO', val)
            self._recover_link()
            time.sleep(1)
            rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
            total = self._total_rate(rates)
            rates_data.append(total)
            self._log(f'  ADC_SYSREF_DEL={val:2d}: rate={total:.4f}', level=1)

        best_idx = self._best_index(rates_data, values, original)

        for adc, oval in originals.items():
            self._adc_set(adc, 'SYSREF_DEL_LO', oval)
        self._adc_disable_all()
        self._recover_link()

        result = {
            'parameter': 'ADC_SYSREF_DEL_LO',
            'values': values, 'rates': rates_data,
            'original': originals[0],
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST: ADC_SYSREF_DEL={result["best_value"]} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    def _sweep_tx_param(self, param_name):
        """Sweep a TX emphasis parameter (txPreCursor or txPostCursor)."""
        values = [0x00, 0x02, 0x04, 0x05, 0x06, 0x08, 0x0A, 0x0C, 0x10, 0x14]

        originals = {}
        for lane in ENABLED_TX_LANES:
            originals[lane] = int(self._tx_get(f'{param_name}[{lane}]'))
        original = originals[ENABLED_TX_LANES[0]]

        rates_data = []
        for val in values:
            for lane in ENABLED_TX_LANES:
                self._tx_set(f'{param_name}[{lane}]', val)
            self._recover_tx_link()
            time.sleep(1)
            rates = self._measure_tx_rate(self.sweep_dwell, label=f"TX sweep pt {val}")
            total = self._total_rate(rates)
            rates_data.append(total)
            self._log(f'  {param_name}=0x{val:02X}: TX rate={total:.4f}', level=1)

        best_idx = self._best_index(rates_data, values, original)

        for lane, oval in originals.items():
            self._tx_set(f'{param_name}[{lane}]', oval)
        self._recover_tx_link()

        result = {
            'parameter': f'TX_{param_name}',
            'values': values, 'rates': rates_data,
            'original': originals[ENABLED_TX_LANES[0]],
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST: {param_name}=0x{result["best_value"]:02X} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    def _sweep_tx_diff_ctrl(self):
        """Sweep TX differential swing control."""
        values = [0x60, 0x80, 0xA0, 0xC0, 0xE0, 0xFF]

        originals = {}
        for lane in ENABLED_TX_LANES:
            originals[lane] = int(self._tx_get(f'txDiffCtrl[{lane}]'))
        original = originals[ENABLED_TX_LANES[0]]

        rates_data = []
        for val in values:
            for lane in ENABLED_TX_LANES:
                self._tx_set(f'txDiffCtrl[{lane}]', val)
            self._recover_tx_link()
            time.sleep(1)
            rates = self._measure_tx_rate(self.sweep_dwell, label=f"TX sweep pt {val}")
            total = self._total_rate(rates)
            rates_data.append(total)
            self._log(f'  txDiffCtrl=0x{val:02X}: TX rate={total:.4f}', level=1)

        best_idx = self._best_index(rates_data, values, original)

        for lane, oval in originals.items():
            self._tx_set(f'txDiffCtrl[{lane}]', oval)
        self._recover_tx_link()

        result = {
            'parameter': 'TX_txDiffCtrl',
            'values': values, 'rates': rates_data,
            'original': originals[ENABLED_TX_LANES[0]],
            'best_value': values[best_idx], 'best_rate': rates_data[best_idx],
        }
        self._log(f'  BEST: txDiffCtrl=0x{result["best_value"]:02X} '
                  f'(rate={result["best_rate"]:.4f})')
        return result

    # =========================================================================
    # Phase 10: Combinatorial Optimization
    # =========================================================================

    def _combinatorial_optimization(self, sweep_results):
        """
        Apply the best value from each independent sweep simultaneously and
        verify they don't conflict (emphasis + swing can interact).
        """
        self._log('\n' + '=' * 70)
        self._log('PHASE 10: COMBINATORIAL OPTIMIZATION')
        self._log('=' * 70)

        # Collect best settings per parameter
        best_settings = {}
        for sr in sweep_results:
            best_settings[sr['parameter']] = sr['best_value']

        self._log('Applying all best settings simultaneously:')
        for param, val in best_settings.items():
            self._log(f'  {param} = 0x{val:X}' if isinstance(val, int) and val > 9
                      else f'  {param} = {val}', level=1)

        # Apply all best settings
        self._apply_settings(best_settings)
        self._recover_link()
        time.sleep(2)

        # Measure combined result
        self._log(f'  Measuring combined performance ({self.sweep_dwell}s)...')
        rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
        combined_rate = self._total_rate(rates)
        self._log(f'  Combined total rate: {combined_rate:.6f} inc/s')

        # Compare to individual bests
        individual_bests = [sr['best_rate'] for sr in sweep_results]
        min_individual = min(individual_bests) if individual_bests else 999

        if combined_rate > min_individual * 2 and combined_rate > 0.01:
            self._log('  WARNING: Combined settings worse than individual bests!')
            self._log('  Likely parameter interaction. Trying reduced combination...')
            # Fall back to just the single best sweep result
            best_single = min(sweep_results, key=lambda x: x['best_rate'])
            reduced = {best_single['parameter']: best_single['best_value']}
            # Add SysrefDelay if it was helpful
            sysref_sweep = next((s for s in sweep_results
                                 if s['parameter'] == 'SysrefDelay'), None)
            if sysref_sweep and sysref_sweep['best_rate'] < sysref_sweep['rates'][
                    sysref_sweep['values'].index(sysref_sweep['original'])
                    if sysref_sweep['original'] in sysref_sweep['values'] else 0]:
                reduced['SysrefDelay'] = sysref_sweep['best_value']

            self._apply_settings(reduced)
            self._recover_link()
            time.sleep(2)
            rates = self._measure_rx_rate(self.sweep_dwell, label=f"Sweep pt {val}")
            combined_rate = self._total_rate(rates)
            best_settings = reduced
            self._log(f'  Reduced combination rate: {combined_rate:.6f} inc/s')

        return {
            'best_settings': best_settings,
            'combined_rate': combined_rate,
            'per_lane': dict(rates),
        }

    def _apply_settings(self, settings):
        """Apply a dict of parameter->value settings."""
        # Enable ADC nodes if any ADC params are being set
        adc_params = ('ADC_SEL_EMP_coarse', 'ADC_SEL_EMP_fine',
                      'ADC_SEL_EMP_full', 'JESD_OUTPUT_SWING', 'ADC_SYSREF_DEL_LO')
        needs_adc = any(p in settings for p in adc_params)
        if needs_adc:
            self._adc_enable_all()

        for param, val in settings.items():
            if param == 'SysrefDelay':
                self._rx_set('SysrefDelay', val)
            elif param in ('ADC_SEL_EMP_coarse', 'ADC_SEL_EMP_fine',
                           'ADC_SEL_EMP_full'):
                for adc in [0, 1]:
                    for ch in [0, 1]:
                        self._adc_ch_set(adc, ch, 'SEL_EMP_LANE0', val)
                        self._adc_ch_set(adc, ch, 'SEL_EMP_LANE2', val)
            elif param == 'JESD_OUTPUT_SWING':
                for adc in [0, 1]:
                    self._adc_set(adc, 'JESD_OUTPUT_SWING', val)
            elif param == 'ADC_SYSREF_DEL_LO':
                for adc in [0, 1]:
                    self._adc_set(adc, 'SYSREF_DEL_LO', val)
            elif param == 'TX_txPreCursor':
                for lane in ENABLED_TX_LANES:
                    self._tx_set(f'txPreCursor[{lane}]', val)
            elif param == 'TX_txPostCursor':
                for lane in ENABLED_TX_LANES:
                    self._tx_set(f'txPostCursor[{lane}]', val)
            elif param == 'TX_txDiffCtrl':
                for lane in ENABLED_TX_LANES:
                    self._tx_set(f'txDiffCtrl[{lane}]', val)

    # =========================================================================
    # Phase 11: Extended Validation
    # =========================================================================

    def _extended_validation(self):
        """Run extended measurement at the applied best settings."""
        self._log('\n' + '=' * 70)
        self._log('PHASE 11: EXTENDED VALIDATION')
        self._log('=' * 70)
        self._log(f'Running {self.validation_dwell}s validation measurement...')

        rates = self._measure_rx_rate(self.validation_dwell, label="Final validation")
        total = self._total_rate(rates)

        self._log(f'Final validated rate: {total:.6f} inc/s')
        for lane in ENABLED_RX_LANES:
            if rates[lane] > 0:
                self._log(f'  Lane {lane}: {rates[lane]:.6f} inc/s (still active)',
                          level=1)

        # Also check TX
        tx_rates = self._measure_tx_rate(dwell=min(60, self.validation_dwell // 5), label="TX validation")
        tx_total = self._total_rate(tx_rates)
        self._log(f'TX validated rate: {tx_total:.6f} inc/s')

        return {
            'rx_rates': dict(rates), 'rx_total': total,
            'tx_rates': dict(tx_rates), 'tx_total': tx_total,
        }

    # =========================================================================
    # Root cause classification
    # =========================================================================

    def _determine_root_cause(self, findings):
        """Synthesize all findings into a root cause classification."""
        iso = findings.get('error_isolation', {})

        if iso.get('gt_or_pll_unstable'):
            return 'gt_pll_instability'

        si_score = 0
        timing_score = 0
        align_score = 0

        per_error = iso.get('per_error', {})
        for err_class, err_names in ERR_CLASS.items():
            for name in err_names:
                if name in per_error and per_error[name].get('triggers_resync'):
                    rate = per_error[name]['total_rate']
                    if err_class == 'signal_integrity':
                        si_score += rate
                    elif err_class == 'timing_buffer':
                        timing_score += rate
                    elif err_class == 'alignment':
                        align_score += rate

        # Weight by corroborating evidence
        sysref = findings.get('sysref', {})
        if not sysref.get('rx_ok', True):
            timing_score *= 3

        el_buf = findings.get('elastic_buffer', {})
        if el_buf.get('risks'):
            timing_score *= 2

        scores = {
            'signal_integrity': si_score,
            'timing_buffer': timing_score,
            'alignment': align_score,
        }

        if max(scores.values()) == 0:
            return 'unknown'

        return max(scores, key=scores.get)

    # =========================================================================
    # Plotting
    # =========================================================================

    def _get_system_id_text(self):
        """Return system identification as a list of strings for reports."""
        lines = [f'Server: {self.S.epics_root}']
        try:
            lines.append(f'Carrier SN: {self.S.get_carrier_sn(use_shell=True)}')
        except Exception:
            pass
        try:
            lines.append(f'Slot: {self.S.slot_number}')
        except Exception:
            pass
        for bay in self.bays:
            try:
                amc_sn = self.S.get_amc_sn(bay=bay, use_shell=True)
                lines.append(f'Bay {bay} AMC: {amc_sn}')
            except Exception:
                pass
        return lines

    def _generate_combined_pdf(self, results):
        """Generate a single PDF report covering all bays."""
        if not HAS_MPL:
            self._log('matplotlib not available; skipping PDF.')
            return

        pdf_path = self.output_dir / 'jesd_tuning_report.pdf'
        self._log(f'\nGenerating combined PDF: {pdf_path}')

        with PdfPages(str(pdf_path)) as pdf:
            # --- Title page ---
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.82, 'JESD204B Link Diagnostic & Tuning Report',
                     ha='center', fontsize=18, fontweight='bold')
            fig.text(0.5, 0.75, datetime.now().strftime('%Y-%m-%d %H:%M'),
                     ha='center', fontsize=12, color='gray')

            # System ID block
            id_lines = self._get_system_id_text()
            y = 0.65
            for line in id_lines:
                fig.text(0.5, y, line, ha='center', fontsize=11,
                         fontfamily='monospace')
                y -= 0.04

            # Per-bay summary
            y -= 0.04
            for bay in self.bays:
                bd = results.get(bay, {})
                cause = bd.get('root_cause_class', 'N/A')
                bl = bd.get('baseline', {}).get('total_rate', 0)
                final = bd.get('validation', {}).get('rx_total', None)
                final_str = f'{final:.4f}' if final is not None else 'N/A'
                fig.text(0.5, y,
                         f'Bay {bay}: {cause} | baseline={bl:.4f} → final={final_str} inc/s',
                         ha='center', fontsize=11, fontfamily='monospace')
                y -= 0.04

            pdf.savefig(fig)
            plt.close()

            # --- Per-bay pages ---
            for bay in self.bays:
                all_data = results.get(bay, {})
                if not all_data:
                    continue

                # Bay header page
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.5, 0.5, f'Bay {bay}', ha='center', fontsize=24,
                         fontweight='bold')
                cause = all_data.get('root_cause_class', 'N/A')
                fig.text(0.5, 0.42, f'Root cause: {cause}', ha='center',
                         fontsize=14, color='darkred')
                try:
                    amc_sn = self.S.get_amc_sn(bay=bay, use_shell=True)
                    fig.text(0.5, 0.36, f'AMC: {amc_sn}', ha='center', fontsize=12)
                except Exception:
                    pass
                pdf.savefig(fig)
                plt.close()

                # Baseline per-lane rates
                baseline = all_data.get('baseline', {})
                if baseline.get('rates'):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    lanes = list(baseline['rates'].keys())
                    rates = [baseline['rates'][l] for l in lanes]
                    colors = ['red' if r > 0 else 'green' for r in rates]
                    ax.bar([str(l) for l in lanes], rates, color=colors)
                    ax.set_xlabel('Lane')
                    ax.set_ylabel('Resync Rate (inc/sec)')
                    ax.set_title(f'Bay {bay} — Baseline Per-Lane Resync Rates '
                                 f'({self.baseline_dwell}s dwell)')
                    ax.grid(True, alpha=0.3, axis='y')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Error isolation
                iso = all_data.get('error_isolation', {}).get('per_error', {})
                if iso:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    names = list(iso.keys())
                    iso_rates = [iso[n]['total_rate'] for n in names]
                    colors = ['red' if r > 0 else 'green' for r in iso_rates]
                    ax.barh(names, iso_rates, color=colors)
                    ax.set_xlabel('Resync Rate (inc/sec)')
                    ax.set_title(f'Bay {bay} — Error Type Isolation')
                    ax.grid(True, alpha=0.3, axis='x')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Sweep plots
                sweeps = all_data.get('sweep_results', [])
                for sweep in sweeps:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    param = sweep['parameter']
                    values = sweep['values']
                    rates = sweep['rates']
                    best_val = sweep['best_value']
                    orig_val = sweep['original']

                    if 'labels' in sweep:
                        x = range(len(values))
                        ax.set_xticks(list(x))
                        ax.set_xticklabels(sweep['labels'], rotation=45, ha='right')
                    elif max(values) > 15:
                        x = range(len(values))
                        ax.set_xticks(list(x))
                        ax.set_xticklabels([f'0x{v:02X}' for v in values],
                                           rotation=45, ha='right')
                    else:
                        x = values

                    ax.plot(list(x), rates, 'b.-', linewidth=2, markersize=8)

                    best_idx = values.index(best_val)
                    bx = list(x)[best_idx]
                    ax.axvline(bx, color='g', linestyle='--', alpha=0.7,
                               label=f'Best: {best_val}')
                    ax.plot(bx, rates[best_idx], 'g*', markersize=15)

                    if orig_val in values:
                        orig_idx = values.index(orig_val)
                        ox = list(x)[orig_idx]
                        ax.axvline(ox, color='r', linestyle=':', alpha=0.7,
                                   label=f'Original: {orig_val}')

                    ax.set_xlabel(param)
                    ax.set_ylabel('Resync Rate (inc/sec)')
                    ax.set_title(f'Bay {bay} — {param} Sweep')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(bottom=0)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # ElBuf heatmap for SysrefDelay
                sysref_sweep = next((s for s in sweeps
                                     if s['parameter'] == 'SysrefDelay'
                                     and 'el_buf_latencies' in s), None)
                if sysref_sweep:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    lat_data = sysref_sweep['el_buf_latencies']
                    matrix = np.zeros((len(ENABLED_RX_LANES), len(lat_data)))
                    for j, lats in enumerate(lat_data):
                        for i, lane in enumerate(ENABLED_RX_LANES):
                            matrix[i, j] = lats.get(lane, 0)
                    im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                                   interpolation='nearest')
                    ax.set_xticks(range(len(sysref_sweep['values'])))
                    ax.set_xticklabels(sysref_sweep['values'])
                    ax.set_yticks(range(len(ENABLED_RX_LANES)))
                    ax.set_yticklabels([f'Lane {l}' for l in ENABLED_RX_LANES])
                    ax.set_xlabel('SysrefDelay')
                    ax.set_title(f'Bay {bay} — Elastic Buffer Latency vs SysrefDelay')
                    plt.colorbar(im, ax=ax, label='Latency (clks)')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Summary bar chart
                if sweeps:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    params = [s['parameter'] for s in sweeps]
                    orig_rates = []
                    best_rates = [s['best_rate'] for s in sweeps]
                    for s in sweeps:
                        if s['original'] in s['values']:
                            orig_idx = s['values'].index(s['original'])
                            orig_rates.append(s['rates'][orig_idx])
                        else:
                            orig_rates.append(s['rates'][0])

                    x_pos = np.arange(len(params))
                    width = 0.35
                    ax.bar(x_pos - width / 2, orig_rates, width,
                           label='Original', color='salmon')
                    ax.bar(x_pos + width / 2, best_rates, width,
                           label='Optimized', color='mediumseagreen')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(params, rotation=30, ha='right')
                    ax.set_ylabel('Resync Rate (inc/sec)')
                    ax.set_title(f'Bay {bay} — Original vs Optimized')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

        self._log(f'  PDF: {pdf_path}')

    def _generate_plots(self, all_data):
        """Generate comprehensive PDF report with all plots."""
        if not HAS_MPL:
            self._log('matplotlib not available; skipping plots.')
            return

        bay = all_data.get('bay', self.bay)
        pdf_path = self.output_dir / f'jesd_tuning_report_bay{bay}.pdf'
        self._log(f'\nGenerating plots: {pdf_path}')

        with PdfPages(str(pdf_path)) as pdf:
            # Title page
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.7, 'JESD204B Link Diagnostic & Tuning Report',
                     ha='center', fontsize=18, fontweight='bold')
            fig.text(0.5, 0.6, f'Bay {self.bay} | {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                     ha='center', fontsize=14)
            fig.text(0.5, 0.5, f'Root cause: {all_data.get("root_cause_class", "N/A")}',
                     ha='center', fontsize=14, color='darkred')
            fig.text(0.5, 0.4, f'Baseline rate: {all_data.get("baseline", {}).get("total_rate", 0):.4f} inc/s',
                     ha='center', fontsize=12)
            final = all_data.get('validation', {}).get('rx_total', None)
            if final is not None:
                fig.text(0.5, 0.35, f'Final optimized rate: {final:.4f} inc/s',
                         ha='center', fontsize=12, color='darkgreen')
            pdf.savefig(fig)
            plt.close()

            # Baseline per-lane rates
            baseline = all_data.get('baseline', {})
            if baseline.get('rates'):
                fig, ax = plt.subplots(figsize=(10, 5))
                lanes = list(baseline['rates'].keys())
                rates = [baseline['rates'][l] for l in lanes]
                colors = ['red' if r > 0 else 'green' for r in rates]
                ax.bar([str(l) for l in lanes], rates, color=colors)
                ax.set_xlabel('Lane')
                ax.set_ylabel('Resync Rate (inc/sec)')
                ax.set_title('Baseline Per-Lane Resync Rates')
                ax.grid(True, alpha=0.3, axis='y')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

            # Error isolation results
            iso = all_data.get('error_isolation', {}).get('per_error', {})
            if iso:
                fig, ax = plt.subplots(figsize=(10, 5))
                names = list(iso.keys())
                iso_rates = [iso[n]['total_rate'] for n in names]
                colors = ['red' if r > 0 else 'green' for r in iso_rates]
                ax.barh(names, iso_rates, color=colors)
                ax.set_xlabel('Resync Rate (inc/sec)')
                ax.set_title('Error Type Isolation (individual enable)')
                ax.grid(True, alpha=0.3, axis='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

            # Sweep plots
            sweeps = all_data.get('sweep_results', [])
            for sweep in sweeps:
                fig, ax = plt.subplots(figsize=(10, 5))
                param = sweep['parameter']
                values = sweep['values']
                rates = sweep['rates']
                best_val = sweep['best_value']
                orig_val = sweep['original']

                if 'labels' in sweep:
                    x = range(len(values))
                    ax.set_xticks(x)
                    ax.set_xticklabels(sweep['labels'], rotation=45, ha='right')
                elif max(values) > 15:
                    x = range(len(values))
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'0x{v:02X}' for v in values],
                                       rotation=45, ha='right')
                else:
                    x = values

                ax.plot(list(x), rates, 'b.-', linewidth=2, markersize=8)

                # Mark best
                best_idx = values.index(best_val)
                bx = list(x)[best_idx]
                ax.axvline(bx, color='g', linestyle='--', alpha=0.7,
                           label=f'Best: {best_val}')
                ax.plot(bx, rates[best_idx], 'g*', markersize=15)

                # Mark original
                if orig_val in values:
                    orig_idx = values.index(orig_val)
                    ox = list(x)[orig_idx]
                    ax.axvline(ox, color='r', linestyle=':', alpha=0.7,
                               label=f'Original: {orig_val}')

                ax.set_xlabel(param)
                ax.set_ylabel('Resync Rate (inc/sec)')
                ax.set_title(f'{param} Sweep')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

            # Elastic buffer heatmap for SysrefDelay sweep
            sysref_sweep = next((s for s in sweeps
                                 if s['parameter'] == 'SysrefDelay'
                                 and 'el_buf_latencies' in s), None)
            if sysref_sweep:
                fig, ax = plt.subplots(figsize=(12, 5))
                lat_data = sysref_sweep['el_buf_latencies']
                matrix = np.zeros((len(ENABLED_RX_LANES), len(lat_data)))
                for j, lats in enumerate(lat_data):
                    for i, lane in enumerate(ENABLED_RX_LANES):
                        matrix[i, j] = lats.get(lane, 0)
                im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                               interpolation='nearest')
                ax.set_xticks(range(len(sysref_sweep['values'])))
                ax.set_xticklabels(sysref_sweep['values'])
                ax.set_yticks(range(len(ENABLED_RX_LANES)))
                ax.set_yticklabels([f'Lane {l}' for l in ENABLED_RX_LANES])
                ax.set_xlabel('SysrefDelay')
                ax.set_title('Elastic Buffer Latency vs SysrefDelay')
                plt.colorbar(im, ax=ax, label='Latency (clks)')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

            # Summary comparison
            if sweeps:
                fig, ax = plt.subplots(figsize=(10, 6))
                params = [s['parameter'] for s in sweeps]
                orig_rates = []
                best_rates = [s['best_rate'] for s in sweeps]
                for s in sweeps:
                    if s['original'] in s['values']:
                        orig_idx = s['values'].index(s['original'])
                        orig_rates.append(s['rates'][orig_idx])
                    else:
                        orig_rates.append(s['rates'][0])

                x = np.arange(len(params))
                width = 0.35
                ax.bar(x - width / 2, orig_rates, width, label='Original',
                       color='salmon')
                ax.bar(x + width / 2, best_rates, width, label='Optimized',
                       color='mediumseagreen')
                ax.set_xticks(x)
                ax.set_xticklabels(params, rotation=30, ha='right')
                ax.set_ylabel('Resync Rate (inc/sec)')
                ax.set_title('Parameter Sweep: Original vs Best')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Also save individual PNGs for quick viewing
        self._save_individual_pngs(all_data)
        self._log(f'  PDF report: {pdf_path}')

    def _save_individual_pngs(self, all_data):
        """Save key plots as individual PNGs."""
        sweeps = all_data.get('sweep_results', [])
        for sweep in sweeps:
            fig, ax = plt.subplots(figsize=(10, 5))
            param = sweep['parameter']
            values = sweep['values']
            rates = sweep['rates']

            if max(values) > 15:
                x = range(len(values))
                ax.set_xticks(x)
                ax.set_xticklabels([f'0x{v:02X}' for v in values],
                                   rotation=45, ha='right')
            else:
                x = values

            ax.plot(list(x), rates, 'b.-', linewidth=2, markersize=8)
            best_idx = values.index(sweep['best_value'])
            ax.plot(list(x)[best_idx], rates[best_idx], 'g*', markersize=15)
            ax.set_xlabel(param)
            ax.set_ylabel('Resync Rate (inc/sec)')
            ax.set_title(f'{param} Sweep')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

            bay = all_data.get('bay', self.bay)
            png_path = self.output_dir / f'sweep_bay{bay}_{param}.png'
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()

    # =========================================================================
    # Reporting
    # =========================================================================

    def _save_full_report(self, all_data):
        """Save comprehensive text and JSON reports."""
        bay = all_data.get('bay', self.bay)

        # Add system ID to the data
        all_data['system_id'] = self._get_system_id_text()

        # JSON (machine-readable)
        json_path = self.output_dir / f'results_bay{bay}.json'

        def json_safe(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2, default=json_safe)

        # Text report
        bay = all_data.get('bay', self.bay)
        txt_path = self.output_dir / f'report_bay{bay}.txt'
        with open(txt_path, 'w') as f:
            f.write('=' * 70 + '\n')
            f.write('JESD204B LINK DIAGNOSTIC AND TUNING REPORT\n')
            f.write(f'Generated: {datetime.now().isoformat()}\n')
            f.write(f'Bay: {bay}\n')
            f.write(f'Baseline dwell: {self.baseline_dwell}s, '
                    f'Diag dwell: {self.diag_dwell}s, '
                    f'Sweep dwell: {self.sweep_dwell}s\n')
            f.write('\nSystem:\n')
            for line in all_data.get('system_id', []):
                f.write(f'  {line}\n')
            f.write('=' * 70 + '\n\n')

            # Root cause
            f.write(f'ROOT CAUSE: {all_data.get("root_cause_class", "N/A")}\n\n')

            # Baseline
            bl = all_data.get('baseline', {})
            f.write('BASELINE RATES (inc/sec):\n')
            for lane, rate in bl.get('rates', {}).items():
                f.write(f'  Lane {lane}: {rate:.6f}\n')
            f.write(f'  Total: {bl.get("total_rate", 0):.6f}\n\n')

            # SYSREF
            sr = all_data.get('sysref', {})
            f.write(f'SYSREF: RX min={sr.get("rx_min")} max={sr.get("rx_max")} '
                    f'[{"OK" if sr.get("rx_ok") else "JITTERY"}]\n')
            f.write(f'        TX min={sr.get("tx_min")} max={sr.get("tx_max")} '
                    f'[{"OK" if sr.get("tx_ok") else "JITTERY"}]\n\n')

            # Elastic buffer
            eb = all_data.get('elastic_buffer', {})
            f.write('ELASTIC BUFFER LATENCY:\n')
            for lane, lat in eb.get('latencies', {}).items():
                f.write(f'  Lane {lane}: {lat} clks\n')
            f.write(f'  Mean={eb.get("mean", 0):.1f}, '
                    f'Std={eb.get("std", 0):.1f}, '
                    f'Spread={eb.get("spread", 0)}\n\n')

            # Error isolation
            iso = all_data.get('error_isolation', {})
            f.write('ERROR TYPE ISOLATION:\n')
            for name, info in iso.get('per_error', {}).items():
                triggers = 'YES' if info['triggers_resync'] else 'no'
                f.write(f'  {name:12s}: triggers={triggers}, '
                        f'rate={info["total_rate"]:.4f}\n')
            f.write('\n')

            # Sweep results
            f.write('SWEEP RESULTS:\n')
            for sweep in all_data.get('sweep_results', []):
                f.write(f'  {sweep["parameter"]}:\n')
                f.write(f'    Original: {sweep["original"]}\n')
                f.write(f'    Best: {sweep["best_value"]} '
                        f'(rate={sweep["best_rate"]:.6f})\n')
                f.write(f'    Values: {sweep["values"]}\n')
                f.write(f'    Rates:  {[f"{r:.4f}" for r in sweep["rates"]]}\n\n')

            # Final settings
            opt = all_data.get('optimization', {})
            f.write('APPLIED OPTIMAL SETTINGS:\n')
            for param, val in opt.get('best_settings', {}).items():
                f.write(f'  {param}: {val}\n')
            f.write(f'\nCombined rate: {opt.get("combined_rate", "N/A")}\n\n')

            # Validation
            val = all_data.get('validation', {})
            f.write(f'FINAL VALIDATION ({self.validation_dwell}s):\n')
            f.write(f'  RX total: {val.get("rx_total", "N/A")}\n')
            f.write(f'  TX total: {val.get("tx_total", "N/A")}\n')

            # Recommendations
            f.write('\n' + '=' * 70 + '\n')
            f.write('RECOMMENDATIONS:\n')
            f.write('=' * 70 + '\n')
            self._write_recommendations(f, all_data)

        self._log(f'Reports saved: {txt_path}, {json_path}')

    def _write_recommendations(self, f, all_data):
        """Generate actionable recommendations."""
        cause = all_data.get('root_cause_class', 'unknown')

        if cause == 'gt_pll_instability':
            f.write('''
- GT/PLL INSTABILITY detected. This is NOT a tunable parameter issue.
- Check: LMK clock distribution, FPGA VCXO, power supply noise
- Measure device clock jitter at FPGA input with spectrum analyzer
- Verify LMK PLL lock status and loop bandwidth settings
- Check for temperature-dependent behavior (CDR can lose lock with drift)
''')
        elif cause == 'signal_integrity':
            f.write('''
- SIGNAL INTEGRITY is the dominant failure mode (disparity/decode errors)
- Apply the optimized ADC emphasis and swing settings from this report
- For persistent issues on new boards, consider:
  - TDR measurement of JESD traces (look for impedance discontinuities)
  - Eye diagram measurement at FPGA receiver (need Xilinx IBERT)
  - Via stub length on new PCB stackup
  - BGA solder joint quality inspection on worst lane(s)
''')
        elif cause == 'timing_buffer':
            f.write('''
- TIMING/BUFFER issues dominate (underflow or overflow)
- Apply the optimized SysrefDelay from this report
- If SYSREF jitter was detected:
  - Check LMK SYSREF output configuration (edge rate, amplitude)
  - Verify SYSREF routing has matched length to all devices
  - Consider increasing SYSREF period (LmkReg_0x0144)
- Monitor ElBuffLatency over time; if it drifts, indicates clock relationship issue
''')
        elif cause == 'alignment':
            f.write('''
- ALIGNMENT errors dominate
- This usually indicates comma (K28.5) detection issues
- Check: ScrambleEnable consistency between ADC and FPGA
- Verify F, K parameters match between all devices
- If only one lane, may be a physical layer comma misalignment issue
''')
        else:
            f.write('''
- Root cause could not be definitively classified
- The optimized settings may still help; apply and monitor
- Consider running with longer dwell times for more statistical confidence
- If issues are very intermittent, temperature/vibration correlation may help
''')

        # Always recommend
        opt = all_data.get('optimization', {})
        if opt.get('best_settings'):
            f.write(f'\nTo apply optimal settings to defaults_c03_lb_lb.yml:\n')
            for param, val in opt['best_settings'].items():
                if param == 'SysrefDelay':
                    f.write(f'  SysrefDelay: 0x{val:X}\n')
                elif 'SEL_EMP' in param:
                    f.write(f'  SEL_EMP_LANE0: 0x{val:X}\n')
                    f.write(f'  SEL_EMP_LANE2: 0x{val:X}\n')
                elif param == 'JESD_OUTPUT_SWING':
                    f.write(f'  JESD_OUTPUT_SWING: 0x{val:X}\n')
                elif param == 'ADC_SYSREF_DEL_LO':
                    f.write(f'  SYSREF_DEL_LO: 0x{val:X}\n')

    # =========================================================================
    # Top-level entry point
    # =========================================================================

    def _run_bay(self, bay):
        """
        Run the complete diagnostic and tuning pipeline for a single bay.
        Returns a dict with all findings and results.
        """
        self._set_bay(bay)
        self._log(f'\n{"#" * 70}')
        self._log(f'  BAY {bay}')
        self._log('#' * 70)
        self._report(f'\n\n{"#" * 70}')
        self._report(f'BAY {bay}')
        self._report('#' * 70)

        all_data = {'bay': bay}

        # Phase 1: System snapshot
        all_data['snapshot'] = self._snapshot_system()
        self._report_phase(f'BAY {bay} PHASE 1: SYSTEM SNAPSHOT', {
            'mismatches': all_data['snapshot'].get('mismatches', []),
            'rx_enable': f"0x{all_data['snapshot']['rx'].get('Enable', 0):X}",
            'tx_enable': f"0x{all_data['snapshot']['tx'].get('Enable', 0):X}",
        })

        # Phase 2: Baseline measurement (10 min default)
        all_data['baseline'] = self._baseline_measurement()
        self._report_phase(f'BAY {bay} PHASE 2: BASELINE', {
            'total_rate': f"{all_data['baseline']['total_rate']:.6f} inc/s",
            'bad_lanes': all_data['baseline']['bad_lanes'],
            'per_lane': {l: f"{r:.6f}" for l, r in all_data['baseline']['rates'].items()},
        })

        if not all_data['baseline']['bad_lanes']:
            self._log('\n*** All lanes stable at baseline. Running sweeps anyway '
                      'to find optimal operating point. ***')
            self._report(f'*** BAY {bay}: BASELINE STABLE — proceeding with '
                         f'optimization sweeps ***')

        # Phase 3: Error classification
        all_data['errors'] = self._classify_errors()
        self._report_phase(f'BAY {bay} PHASE 3: ERROR CLASSIFICATION',
                           all_data['errors'].get('error_types', {}))

        # Phase 4: SYSREF quality
        all_data['sysref'] = self._check_sysref()
        self._report_phase(f'BAY {bay} PHASE 4: SYSREF', all_data['sysref'])

        # Phase 5: Elastic buffer analysis
        all_data['elastic_buffer'] = self._analyze_elastic_buffer()
        self._report_phase(f'BAY {bay} PHASE 5: ELASTIC BUFFER', {
            'latencies': all_data['elastic_buffer']['latencies'],
            'mean': f"{all_data['elastic_buffer']['mean']:.1f}",
            'spread': all_data['elastic_buffer']['spread'],
            'risks': all_data['elastic_buffer']['risks'],
        })

        # Phase 6: Link error isolation (skip if baseline was clean)
        if all_data['baseline']['bad_lanes']:
            all_data['error_isolation'] = self._isolate_link_errors()
            iso_summary = {}
            if not all_data['error_isolation'].get('gt_or_pll_unstable'):
                for name, info in all_data['error_isolation'].get('per_error', {}).items():
                    iso_summary[name] = f"{'TRIGGERS' if info['triggers_resync'] else 'stable'} ({info['total_rate']:.4f})"
            else:
                iso_summary['GT/PLL'] = 'UNSTABLE — links drop even with all errors masked'
            self._report_phase(f'BAY {bay} PHASE 6: ERROR ISOLATION', iso_summary)
        else:
            all_data['error_isolation'] = {}
            self._report_phase(f'BAY {bay} PHASE 6: ERROR ISOLATION',
                               {'skipped': 'baseline clean'})

        # Phase 7: Per-lane isolation (skip if baseline was clean)
        if all_data['baseline']['bad_lanes']:
            all_data['lane_isolation'] = self._isolate_lanes(
                all_data['baseline']['bad_lanes'])
            self._report_phase(f'BAY {bay} PHASE 7: LANE ISOLATION', all_data['lane_isolation'])
        else:
            all_data['lane_isolation'] = {}
            self._report_phase(f'BAY {bay} PHASE 7: LANE ISOLATION',
                               {'skipped': 'baseline clean'})

        # Phase 8: TX link health
        all_data['tx_health'] = self._check_tx_health()
        self._report_phase(f'BAY {bay} PHASE 8: TX HEALTH', {
            'has_issues': all_data['tx_health']['has_issues'],
            'total_rate': f"{all_data['tx_health']['total_rate']:.4f} inc/s",
        })

        # Determine root cause
        all_data['root_cause_class'] = self._determine_root_cause(all_data)
        all_data['tx_has_issues'] = all_data['tx_health']['has_issues']
        self._log(f'\n>>> BAY {bay} ROOT CAUSE: {all_data["root_cause_class"]}')
        self._report(f'\n>>> BAY {bay} ROOT CAUSE: {all_data["root_cause_class"]}')

        # Phase 9: Adaptive sweeps
        sweep_plan = self._plan_sweeps(all_data)
        all_data['sweep_results'] = self._run_sweeps(sweep_plan, all_data)
        for sr in all_data['sweep_results']:
            self._report_phase(f'BAY {bay} SWEEP: {sr["parameter"]}', {
                'original': sr['original'],
                'best_value': sr['best_value'],
                'best_rate': f"{sr['best_rate']:.6f} inc/s",
                'all_rates': [f"{r:.4f}" for r in sr['rates']],
            })

        # Phase 10: Combinatorial optimization
        if all_data['sweep_results']:
            all_data['optimization'] = self._combinatorial_optimization(
                all_data['sweep_results'])
        else:
            all_data['optimization'] = {}
        self._report_phase(f'BAY {bay} PHASE 10: OPTIMIZATION', {
            'settings': all_data['optimization'].get('best_settings', {}),
            'combined_rate': all_data['optimization'].get('combined_rate', 'N/A'),
        })

        # Phase 11: Extended validation
        all_data['validation'] = self._extended_validation()
        self._report_phase(f'BAY {bay} PHASE 11: VALIDATION', {
            'rx_total': f"{all_data['validation']['rx_total']:.6f} inc/s",
            'tx_total': f"{all_data['validation']['tx_total']:.6f} inc/s",
        })

        # Ensure ADC nodes are disabled when done with this bay
        self._adc_disable_all()

        return all_data

    def run(self):
        """
        Run the complete diagnostic and tuning pipeline for all bays.
        Returns a dict keyed by bay number with all findings and results.
        """
        start_time = datetime.now()
        self._log('=' * 70)
        self._log(f'JESD204B COMPREHENSIVE DIAGNOSTIC AND TUNING')
        self._log(f'Started: {start_time}')
        self._log(f'Output: {self.output_dir}/')
        self._log(f'Bays: {self.bays}')
        self._log(f'Timing: baseline={self.baseline_dwell}s, '
                  f'diag={self.diag_dwell}s, sweep={self.sweep_dwell}s, '
                  f'validation={self.validation_dwell}s')
        self._log('=' * 70)

        results = {}
        for bay in self.bays:
            results[bay] = self._run_bay(bay)

        # Final summary
        self._report(f'\n{"=" * 70}')
        self._report('COMPLETE — ALL BAYS')
        elapsed = (datetime.now() - start_time).total_seconds()
        self._report(f'Elapsed: {elapsed / 60:.1f} min')
        for bay in self.bays:
            bl_rate = results[bay].get('baseline', {}).get('total_rate', 0)
            final_rate = results[bay].get('validation', {}).get('rx_total', None)
            cause = results[bay].get('root_cause_class', 'N/A')
            final_str = f'{final_rate:.4f}' if final_rate is not None else 'N/A'
            self._report(f'  Bay {bay}: cause={cause}, '
                         f'baseline={bl_rate:.4f}, '
                         f'final={final_str}')
        self._report_file.close()

        # Save per-bay text/JSON reports
        for bay in self.bays:
            self._set_bay(bay)
            self._save_full_report(results[bay])

        # Generate single combined PDF with all bays
        self._generate_combined_pdf(results)

        # Console summary
        self._log('\n' + '=' * 70)
        self._log('COMPLETE — ALL BAYS')
        self._log('=' * 70)
        self._log(f'Elapsed: {elapsed / 60:.1f} minutes')
        for bay in self.bays:
            bl_rate = results[bay].get('baseline', {}).get('total_rate', 0)
            final_rate = results[bay].get('validation', {}).get('rx_total', None)
            cause = results[bay].get('root_cause_class', 'N/A')
            if bl_rate > 0 and final_rate is not None:
                improvement = (1 - final_rate / bl_rate) * 100
                self._log(f'  Bay {bay}: {cause} | '
                          f'{bl_rate:.4f} -> {final_rate:.4f} inc/s '
                          f'({improvement:.1f}% improvement)')
            else:
                self._log(f'  Bay {bay}: {cause} | baseline={bl_rate:.4f} inc/s')
        self._log(f'Reports: {self.output_dir}/')

        return results


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='JESD204B Comprehensive Link Diagnostic and Tuning')
    parser.add_argument('--bays', type=int, nargs='+', default=[0, 1],
                        help='AMC bays to diagnose (default: 0 1)')
    parser.add_argument('--baseline-dwell', type=int, default=600,
                        help='Baseline measurement dwell (sec, default 600)')
    parser.add_argument('--diag-dwell', type=int, default=60,
                        help='Diagnostic measurement dwell (sec, default 60)')
    parser.add_argument('--sweep-dwell', type=int, default=120,
                        help='Per-sweep-point dwell (sec, default 120)')
    parser.add_argument('--validation-dwell', type=int, default=300,
                        help='Final validation dwell (sec, default 300)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            S = ip.user_ns.get('S')
        else:
            S = None
    except ImportError:
        S = None

    if S is None:
        print('ERROR: No pysmurf SmurfControl instance (S) found.')
        print('Run this from a pysmurf IPython session:')
        print('  diag = JesdDiagTune(S, bay=0)')
        print('  results = diag.run()')
        raise SystemExit(1)

    diag = JesdDiagTune(
        S, bays=args.bays,
        output_dir=args.output_dir,
        baseline_dwell=args.baseline_dwell,
        diag_dwell=args.diag_dwell,
        sweep_dwell=args.sweep_dwell,
        extended_validation_dwell=args.validation_dwell,
    )
    results = diag.run()
