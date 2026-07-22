"""
Minimal example: streaming I/Q data with the PreSpectra firmware (v0x04).

WHY THE EXISTING NOTEBOOK GIVES ZEROS:
1. The remap LUT (U_Remap_readAddr) defaults to all zeros at power-on.
   All 64 streaming slots read physical channel 0, which has no tone.
2. The IQ_mode=True flag in take_stream_data/read_stream_data assumes the
   v0x03 data layout.  In v0x04, every int16 in the payload is already a
   single I or Q sample (packed [I, Q, I, Q, ...] across slots and time
   slices), so the old "pair consecutive int32s" logic is wrong.

FIX:
  - Configure the remap LUT before streaming.
  - Don't use IQ_mode=True.  Stream all 4096 positions and parse v0x04 layout.

DATA FORMAT (v0x04):
  4096 int16 values per frame, organized as:
    32 time slices × 64 slots × 2 (I then Q for each slot)
  On disk after ChannelMapper: 4096 int32 values (sign-extended int16s).

FRAME RATE:
  Each trigger (flux ramp strobe) captures ONE time slice (64 slots).
  32 triggers are aggregated into one frame.
  Frame rate = flux_ramp_rate / 32.
  At 32 kHz flux ramp → ~1000 frames/s.

REMAP LUT FORMULA:
  For channel `ch` in band `b`:  remap_value = b * 512 + ch
  For band 0: remap_value = ch  (channels 0-511)
"""

import numpy as np
import time
import os


# ===========================================================================
# 1. REMAP LUT CONFIGURATION
# ===========================================================================
def configure_remap_lut(S, channels, band=0):
    """
    Write the remap LUT so streaming slots map to physical OPFB channels.

    Parameters
    ----------
    S : pysmurf.client.SmurfControl
    channels : array-like of int
        OPFB channel numbers within `band` to stream (max 64).
    band : int
        Band number (0-7). Determines the upper bits of the remap address.
    """
    channels = np.atleast_1d(channels).astype(int)
    if len(channels) > 64:
        raise ValueError("Max 64 channels per frame")

    for slot in range(64):
        val = int(band * 512 + channels[slot]) if slot < len(channels) else 0
        S._caput(f'{S.app_core}RemapReadAddr.RemapLut[{slot}]', val)

    print(f"Remap LUT: slots 0-{len(channels)-1} → channels {channels}")


def read_remap_lut(S, n_slots=64):
    """Read back current remap LUT values."""
    return np.array([
        S._caget(f'{S.app_core}RemapReadAddr.RemapLut[{i}]')
        for i in range(n_slots)
    ])


# ===========================================================================
# 2. TAKE DATA (using existing pysmurf infrastructure)
# ===========================================================================
def take_prespectra_data(S, meas_time, downsample_factor=None):
    """
    Take PreSpectra streaming data using the standard pysmurf data path.

    Streams all 4096 payload positions (= full 64ch × 32 timeslices × I/Q).
    Does NOT use IQ_mode since that flag activates v0x03-specific logic.
    """
    if downsample_factor is not None:
        S.set_downsample_factor(downsample_factor)
    else:
        downsample_factor = S.get_downsample_factor()
        print(f"Using existing downsample_factor={downsample_factor}")

    # Stream ALL 4096 positions — the full v0x04 payload
    mask = list(range(4096))
    S.set_channel_mask(mask)
    S.set_payload_size(0)  # auto-adjust

    time.sleep(0.5)

    # Start streaming
    S.set_stream_enable(1)
    S.set_unwrapper_reset()
    S.set_filter_reset()
    time.sleep(0.15)

    # Open data file
    timestamp = S.get_timestamp()
    data_filename = os.path.join(S.output_dir, timestamp + '.dat')
    S.set_data_file_name(data_filename)
    S.open_data_file()
    print(f"Writing to: {data_filename}")

    time.sleep(meas_time)

    # Close
    S.close_data_file()
    S.set_stream_enable(0)
    print(f"Done.  {data_filename}")

    return data_filename


# ===========================================================================
# 3. READ + PARSE v0x04 FORMAT
# ===========================================================================
def read_prespectra_data(datafile, n_slots=64, n_timeslices=32):
    """
    Parse a PreSpectra v0x04 data file.

    On-disk layout per frame: 4096 int32 values (sign-extended int16).
    Order: for each time slice (0-31), for each slot (0-63): I, Q.

    Returns
    -------
    timestamps : ndarray shape (n_frames,)
    iq : ndarray complex64 shape (n_frames, n_timeslices, n_slots)
        Raw ADC counts.  Divide by 2**14 for Fixed16_14 physical units.
    """
    from pysmurf.client.util.SmurfFileReader import SmurfStreamReader

    expect = n_slots * n_timeslices * 2  # 4096

    timestamps = []
    frames = []

    with SmurfStreamReader(datafile, isRogue=True, metaEnable=False) as reader:
        for header, data in reader.records():
            timestamps.append(header.timestamp)

            if len(data) < expect:
                padded = np.zeros(expect, dtype=np.int32)
                padded[:len(data)] = data
                data = padded
            elif len(data) > expect:
                data = data[:expect]

            # Reshape: (32 timeslices, 64 slots, 2=[I, Q])
            arr = data.reshape(n_timeslices, n_slots, 2)
            iq = arr[:, :, 0].astype(np.float32) + 1j * arr[:, :, 1].astype(np.float32)
            frames.append(iq)

    timestamps = np.array(timestamps)
    iq = np.array(frames)  # (n_frames, 32, 64)
    print(f"Read {len(frames)} frames, shape {iq.shape} = (frames, timeslices, slots)")
    return timestamps, iq


# ===========================================================================
# FULL USAGE EXAMPLE (paste into notebook cells)
# ===========================================================================
EXAMPLE = """
# ── Cell 1: Setup (same as your working notebook) ──────────────────────────
import pysmurf.client
import numpy as np
import matplotlib.pyplot as plt
from prespectra_streaming_example import (
    configure_remap_lut, read_remap_lut,
    take_prespectra_data, read_prespectra_data,
)

port = 9009
config_file = "/usr/local/src/pysmurf/cfg_files/b33/experiment_b33_rfc1-2_C05-40_s4_rfsoc_extref.cfg"
S = pysmurf.client.SmurfControl(server_port=port, cfg_file=config_file,
                                setup=False, make_logfile=False)
S.setup()

band = 0
S.set_band_center_mhz(band, 750)
S.set_timing_mode('ext_ref')
S.set_lms_gain(band, 7)
S.band_off(band)

S.set_fixed_tone(511.101, 12)
S.set_fixed_tone(575.292, 12)
S.set_fixed_tone(636.383, 12)
S.set_fixed_tone(694.474, 12)
S.set_fixed_tone(728.565, 12)
S.set_fixed_tone(755.656, 12)
S.set_fixed_tone(812.747, 12)
S.set_fixed_tone(879.838, 12)
S.set_fixed_tone(931.929, 12)
S.set_fixed_tone(989.010, 12)

active_channels = S.which_on(band)
print(f"Active channels: {active_channels}")

# ── Cell 2: Configure remap LUT (THIS IS THE KEY MISSING STEP) ─────────────
configure_remap_lut(S, active_channels, band=0)
print("Readback:", read_remap_lut(S, n_slots=len(active_channels)))

# ── Cell 3: Flux ramp + filter (same as before) ────────────────────────────
S.stream_data_off()
S.flux_ramp_setup(32, 0)
S.set_filter_disable(False)
S.set_downsample_factor(1)

# ── Cell 4: Take data ──────────────────────────────────────────────────────
data_file = take_prespectra_data(S, meas_time=1)

# ── Cell 5: Read + plot ────────────────────────────────────────────────────
timestamps, iq = read_prespectra_data(data_file)
# iq shape: (n_frames, 32 timeslices, 64 slots)
# Slots 0-9 have our 10 channels; slots 10-63 are channel 0 (no tone).
# Scale to physical: iq / 2**14

n_active = len(active_channels)
iq_phys = iq[:, :, :n_active] / 2**14  # (frames, 32, 10)

# Plot time-slice 0 for all active channels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
offset = 0.2
for i, ch in enumerate(active_channels):
    ts = iq_phys[:, 0, i]  # timeslice 0, slot i, all frames
    ax1.plot(np.real(ts - np.median(np.real(ts))) + offset*i, label=f'ch{ch}')
    ax2.plot(np.imag(ts - np.median(np.imag(ts))) + offset*i, label=f'ch{ch}')

ax1.set_xlabel('Frame #'); ax1.set_ylabel('I'); ax1.legend(fontsize=7)
ax2.set_xlabel('Frame #'); ax2.set_ylabel('Q'); ax2.legend(fontsize=7)
fig.suptitle(f'PreSpectra I/Q — timeslice 0 — {data_file}')
plt.tight_layout()
plt.show()

# ── Cell 6: Use all 32 time slices for one channel (high time resolution) ──
slot = 0  # first active channel
ch_name = active_channels[slot]
# Each frame has 32 samples; concatenate across frames for full time series
full_ts = iq_phys[:, :, slot].flatten()  # interleave all timeslices
print(f"Channel {ch_name}: {len(full_ts)} samples at {32 * 1000} Hz effective rate")

plt.figure(figsize=(10, 3))
plt.plot(np.real(full_ts[:3000]), label='I')
plt.plot(np.imag(full_ts[:3000]), label='Q', alpha=0.7)
plt.xlabel('Sample'); plt.ylabel('Amplitude')
plt.title(f'Channel {ch_name} — 32 kHz effective rate (32 timeslices × frame rate)')
plt.legend()
plt.tight_layout()
plt.show()
"""

if __name__ == '__main__':
    print(EXAMPLE)
