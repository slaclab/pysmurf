"""
Verification script for DSPv3 channel mapping bug.

This script identifies the mismatch between the firmware's actual
processed channel set and pysmurf's computed processed channel set.

The bug: pysmurf uses a symmetric frequency cut (|freq| < Fdsp/2) to
determine which 416 of 512 channels are processed, but the firmware
uses an asymmetric condition (-Fdsp/2 <= freq < Fdsp/2). This causes
a one-channel discrepancy at the bandwidth boundary that shifts the
debug data mapping for channels 23 and 24 (subbands 51 and 75).

Usage:
    Run in a pysmurf session with an active band:
        %run verify_channel_mapping.py

    Or import and call:
        from verify_channel_mapping import verify_channel_mapping
        verify_channel_mapping(S, band=0)

    Set PROBE_FIRMWARE=True to empirically measure the firmware's
    processed set (takes ~10 minutes). Set False to use the theoretical
    computation only.
"""

import numpy as np
import time

PROBE_FIRMWARE = False  # Set True to empirically probe each channel


def get_firmware_processed_empirical(S, band, nsamp=2**12):
    """Probe each channel to find which ones produce non-zero debug data.

    Turns on each channel individually with rf_iq debug data and checks
    for non-zero output. This gives ground truth for which channels the
    firmware actually processes.

    WARNING: This takes ~10 minutes and will disturb any active channels.
    """
    print("Probing firmware processed channels empirically...")
    print("(This turns on each channel individually — will take several minutes)")

    n_channels = S.get_number_channels(band)
    firmware_processed = []
    firmware_unprocessed = []

    for chan in range(n_channels):
        S.set_amplitude_scale_channel(band, chan, 12)
        i, q, sync = S.take_debug_data(band=band, channel=None,
                                        rf_iq=True, nsamp=nsamp)
        S.channel_off(band, chan)

        if len(np.nonzero(i)[0]) > 0:
            firmware_processed.append(chan)
        else:
            firmware_unprocessed.append(chan)

        if (chan + 1) % 64 == 0:
            print(f"  ... probed {chan + 1}/{n_channels} channels")

    return np.array(sorted(firmware_processed)), np.array(sorted(firmware_unprocessed))


def get_firmware_processed_theoretical(S, band):
    """Compute the firmware's processed channel set from first principles.

    The firmware processes the N channels nearest to DC (where N comes
    from get_number_processed_channels). At the bandwidth boundary, the
    firmware includes the negative frequency and excludes the positive
    (asymmetric: lower bound inclusive, upper bound exclusive).
    """
    tone_freq_offset = S.get_tone_frequency_offset_mhz(band)
    n_proc = S.get_number_processed_channels(band)

    # Select the n_proc channels nearest to DC.
    # Tie-break: at equal |freq|, negative is included, positive excluded.
    sort_key = np.abs(tone_freq_offset) + (tone_freq_offset > 0) * 1e-10
    processed = np.sort(np.argsort(sort_key)[:n_proc])
    all_chans = np.arange(len(tone_freq_offset))
    unprocessed = np.sort(np.setdiff1d(all_chans, processed))

    return processed, unprocessed


def get_pysmurf_processed(S, band):
    """Get pysmurf's current computed processed channel set."""
    return S.get_processed_channels()


def analyze_mapping_shift(fw_processed, py_processed, tone_freq_offset):
    """Analyze how the mismatch shifts the debug data mapping.

    Returns list of (stream_position, fw_channel, pysmurf_channel) tuples
    where the mapping diverges.
    """
    fw_order = sorted(fw_processed)
    py_order = sorted(py_processed)

    shifts = []
    n = min(len(fw_order), len(py_order))
    for k in range(n):
        if fw_order[k] != py_order[k]:
            shifts.append((k, fw_order[k], py_order[k]))

    return shifts


def verify_channel_mapping(S, band=0):
    """Run the full verification.

    Args
    ----
    S : pysmurf SmurfControl instance
    band : int
        Which band to verify.

    Returns
    -------
    bool
        True if mapping is correct (no mismatch), False if bug is present.
    """
    print("=" * 70)
    print(f"  Channel Mapping Verification — Band {band}")
    print("=" * 70)

    # Get tone frequency offsets
    tone_freq_offset = S.get_tone_frequency_offset_mhz(band)
    digitizer_frequency_mhz = S.get_digitizer_frequency_mhz(band)
    n_channels = S.get_number_channels(band)
    half_dsp_bw = 0.8125 * digitizer_frequency_mhz / 2

    print(f"\nSystem parameters:")
    print(f"  Digitizer frequency: {digitizer_frequency_mhz} MHz")
    print(f"  DSP bandwidth:       {2 * half_dsp_bw} MHz")
    print(f"  Half DSP BW:         +/- {half_dsp_bw} MHz")
    print(f"  Number of channels:  {n_channels}")
    print(f"  Expected processed:  {int(0.8125 * n_channels)}")

    # --- Firmware processed set ---
    if PROBE_FIRMWARE:
        fw_processed, fw_unprocessed = get_firmware_processed_empirical(S, band)
    else:
        fw_processed, fw_unprocessed = get_firmware_processed_theoretical(S, band)
        print(f"\n  (Using theoretical firmware model — set PROBE_FIRMWARE=True")
        print(f"   to empirically verify with hardware)")

    # --- pysmurf processed set ---
    py_processed = get_pysmurf_processed(S, band)

    print(f"\nProcessed channel counts:")
    print(f"  Firmware:  {len(fw_processed)} channels")
    print(f"  pysmurf:   {len(py_processed)} channels")

    # --- Compare ---
    fw_set = set(fw_processed)
    py_set = set(py_processed)

    only_in_fw = sorted(fw_set - py_set)
    only_in_py = sorted(py_set - fw_set)

    if len(only_in_fw) == 0 and len(only_in_py) == 0:
        print(f"\n*** PASS: Firmware and pysmurf processed sets MATCH perfectly ***")
        print(f"    The channel mapping bug is NOT present (or has been fixed).")
        return True

    print(f"\n*** MISMATCH DETECTED ***")
    print(f"\n  Channels firmware processes but pysmurf excludes ({len(only_in_fw)}):")
    for c in only_in_fw:
        print(f"    Channel {c:3d}  (freq = {tone_freq_offset[c]:+.1f} MHz)")

    print(f"\n  Channels pysmurf includes but firmware does NOT process ({len(only_in_py)}):")
    for c in only_in_py:
        print(f"    Channel {c:3d}  (freq = {tone_freq_offset[c]:+.1f} MHz)")

    # --- Analyze mapping shift ---
    shifts = analyze_mapping_shift(fw_processed, py_processed, tone_freq_offset)

    if shifts:
        print(f"\n  Debug stream mapping errors ({len(shifts)} positions affected):")
        print(f"  {'Stream Pos':<12} {'FW outputs':<20} {'pysmurf assigns to':<20}")
        print(f"  {'-'*12} {'-'*20} {'-'*20}")
        for pos, fw_ch, py_ch in shifts:
            print(f"  {pos:<12d} ch {fw_ch:3d} ({tone_freq_offset[fw_ch]:+.1f} MHz)"
                  f"   ch {py_ch:3d} ({tone_freq_offset[py_ch]:+.1f} MHz)")

        # Check impact on subbands 51 and 75 (channels 23, 24)
        print(f"\n  Impact on subbands 51 and 75:")
        for target_ch, sb_name in [(23, "subband 51 (-57.6 MHz)"),
                                    (24, "subband 75 (+57.6 MHz)")]:
            affected = [(pos, fw_ch, py_ch) for pos, fw_ch, py_ch in shifts
                        if py_ch == target_ch]
            if affected:
                pos, fw_ch, py_ch = affected[0]
                print(f"    Channel {target_ch} ({sb_name}):")
                print(f"      pysmurf reads stream position {pos}")
                print(f"      but that position has data for channel {fw_ch} "
                      f"(freq={tone_freq_offset[fw_ch]:+.1f} MHz)")
                if abs(tone_freq_offset[fw_ch]) > half_dsp_bw - 5:
                    print(f"      -> Band-edge channel! Likely shows FLAT LINE "
                          f"(no resonator)")
                else:
                    print(f"      -> Gets WRONG channel's data")
            else:
                print(f"    Channel {target_ch} ({sb_name}): not directly affected")

    print(f"\n*** FAIL: Channel mapping bug IS present ***")
    print(f"    Fix get_processed_channels() to use asymmetric boundary condition:")
    print(f"    -Fdsp/2 <= freq < Fdsp/2  (not symmetric |freq| < Fdsp/2)")
    return False


if __name__ == "__main__":
    # When run directly in a pysmurf session, 'S' should already be defined
    try:
        S
    except NameError:
        print("ERROR: No pysmurf instance 'S' found.")
        print("Run this script from within an active pysmurf session,")
        print("or set S = your SmurfControl instance before running.")
        raise SystemExit(1)

    band = 0  # Change this to your active band
    result = verify_channel_mapping(S, band=band)
    if result:
        print("\nChannel mapping is correct.")
    else:
        print("\nChannel mapping has errors — see above for details.")
