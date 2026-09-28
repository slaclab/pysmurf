# sodetlib changes required by the register-layer confinement

**Repository:** `simonsobs/sodetlib`, branch `tpm/rogue6`, measured at `04cfa27`.
**Status:** required before or with the pysmurf release that removes the handles below.

## Why this exists

pysmurf's client used to hand out register paths: `S._cryo_root(band)` returned a path prefix,
`S.rtm_spi_max_root` was one, and `_caget`/`_caput` took whatever string was built from them. sodetlib
used that at four sites. Register paths now live only in `cryodaq.platform`, so those handles are gone
from pysmurf and the four sites need somewhere else to go.

The ordering matters: **pysmurf has already removed the handles.** A sodetlib that still reaches for
them raises `AttributeError` on the affected call, so the sodetlib change lands before or with the
pysmurf release that ships this. `tests/cryodaq/check_sodetlib_contract.py` records each dropped name
with its replacement, and fails if one quietly returns, so the removal is a decision on record rather
than a surprise. Supporting sodetlib is required; preserving its access to pysmurf's *private* surface
is not.

## The four sites

### 1. `operations/uxm_relock.py:97-125` — the eta-scan mutex

```python
in_progress_reg = S._cryo_root(b) + 'etaScanInProgress'
if S._caget(in_progress_reg):        # read
    S._caput(in_progress_reg, 0)     # clear, on the force_run path and on error
```

**Replace with** the public pair:

```python
if S.get_eta_scan_in_progress(b):
    S.set_eta_scan_in_progress(b, 0)
```

`get_eta_scan_in_progress` already existed and reads the identical register.
`set_eta_scan_in_progress` is new, added for this: the flag is the server's to raise, but a process
that dies mid-scan leaves it raised and every later scan refuses to start, so clearing it is a real
operation and now has a name. Three call sites in this file (`:98`, `:100`, `:125`).

### 2. `util.py:676-686` — `set_current_mode`, the one real gap

```python
dac_data_reg = S.rtm_spi_max_root + S._rtm_slow_dac_data_reg
...
S._caput(dac_data_reg, dac_data, wait_done=False)   # to high current
S._caput(dac_data_reg, dac_data)                    # to low current
```

**This one is not a re-spelling, and should not be replaced by one.** `set_rtm_slow_dac_data_array`
writes the identical register and even duplicates sodetlib's clipping, and
`set_rtm_slow_dac_volt_array` does the volt conversion — but the bypass is not about those. Its
docstring records a *physical* constraint: set both in a single call to minimise heating on the
cryostat, with deliberately different ordering per direction — to high current, fire-and-forget then
flip the relay; to low current, block, sleep 40 ms, then flip, or the detectors latch.

pysmurf's `set_tes_bias_high_current` flips the relay and never touches the DACs, so there is no public
operation that does both in the required order. **The deliverable is that combined operation**, a
`set_current_mode` in pysmurf. Until it exists this site has nothing to move to, which is why
`rtm_spi_max_root` and `_rtm_slow_dac_data_reg` are recorded as *deliberately dropped* rather than
replaced. 11 call sites in sodetlib depend on this function.

Where the operation belongs is a design question, not a mechanical one: it sequences two registers and a
relay against a thermal constraint, which makes it an operation rather than an accessor.

### 3. `util.py:210-211` — two version reads

```python
'rogue_version': S._caget('AMCc.RogueVersion'),
'smurf_core_version': S._caget('AMCc.SmurfApplication.SmurfVersion'),
```

**Replace with** `S.get_rogue_version()` and `S.get_pysmurf_version()`, which read the identical paths
and have existed all along.

### 4. `util.py:520-583` — the `_Register` / `Registers` wrapper

```python
class _Register:
    def get(self, **kw): return self.S._caget(self.addr, **kw)
    def set(self, val, **kw): self.S._caput(self.addr, val, **kw)
```

Fourteen hardcoded addresses behind a generic wrapper. **Thirteen of the fourteen are not pysmurf's
registers** — `SmurfProcessor.SOStream.*` and `SOFileWriter.*` are `smurf-streamer`'s nodes, as the
class docstring says ("even if they are not in the standard rogue tree"). Measured: zero occurrences of
`SOStream` in pysmurf, and zero in the released firmware package.

So this is not a pysmurf boundary violation and does not have to move into pysmurf's map. The
registers belong to the streamer, and a map for them is SO's to write. Two changes are worth making:

- `_Register.get`/`set` reach `S._caget`/`S._caput` to talk to *non-pysmurf* nodes, i.e. they borrow
  pysmurf's client as a generic rogue transport. That should be a rogue client of its own, or
  `cryodaq.connect`, which is what it is for.
- **The 14th, `source_enable`**, *is* pysmurf's: `AMCc.StreamDataSource.SourceEnable` duplicates the
  existing public `set_stream_data_source_enable` / `get_stream_data_source_enable`. Move that one.

`_caget`/`_caput` still accept a raw path, so this site keeps working for now. Once it has moved, they
can stop accepting one — the last step that makes the boundary hold from both sides.

## Summary

| site | change | difficulty |
|---|---|---|
| `uxm_relock.py:97-125` | `get/set_eta_scan_in_progress(b)` | mechanical; the setter exists |
| `util.py:210-211` | `get_rogue_version()`, `get_pysmurf_version()` | trivial |
| `util.py:676` (`set_current_mode`) | **needs a new combined pysmurf operation** | the one real gap; a design question |
| `util.py:520-583` (`Registers`) | own transport; move `source_enable` only | mostly SO's own namespace, not pysmurf's |

Three of four are straightforward. The fourth is a genuine missing operation, and it is the reason the
private surface is frozen and recorded rather than simply deleted.
