# Stage 2 — client interface specification (draft)

**Status:** draft for review by SLAC (pysmurf / cryo-det maintainers), SO (sodetlib / OCS /
smurf-streamer maintainers) and SPECTRA. **Date:** 2026-09-10, **revised 2026-09-11**. **Author:**
Tristan Pinsonneault-Marotte.
**Derives from:** the refactor proposal's §"The core API (draft)" and the client-interface kick-off
record, whose seven decisions this document assumes and does not re-argue. Both are development
records of the refactor and are kept with it rather than in this repository, as are the `scripts/`,
`artifacts/` and `old_workspace/` paths this document cites as its evidence.

> **Revision 2 (2026-09-11) — the client is thinner than revision 1 described.** The first
> implementation of §§2–4 met these shapes and was green on both firmware trees, and it took 2,300
> lines across nine modules to do it: a client-side name syntax with an entry/catalog object model, a
> declaration-and-handle layer over rogue's own `Process`, a platform base class carrying
> `raw_get`/`raw_set`/`process_start`, typed capability records, and a second in-process route for
> emulation beside the network one. Three wrappers over `getNode().get()`, in a refactor whose reason
> for existing is to remove that. Reviewed and rejected on those grounds; the sections below now
> describe what replaced it. What changed, in one line each:
>
> - **One transport.** An emulated system is a rogue tree served over the same sockets as a real one,
>   so `connect()` takes an endpoint and nothing else, and the emulated route is gone (§2.1).
> - **The client is a rogue client.** `session.root` is the server's tree, unwrapped, and operations are
>   rogue nodes rather than objects of ours. pyrogue is a runtime dependency of `connect()`, its one use
>   site, and of nothing else here — importing the package, reading a map, building paths and parsing an
>   endpoint all work without it (§2.3).
> - **Operations are rogue nodes.** No `OperationSpec`, `Provider`, `Param`, `Result` or `Handle`:
>   `call()` starts a command or a process and the caller reads the process itself (§4.1).
> - **The platform is a map and nothing else.** Per-platform data — paths, names, witness set, how to
>   enumerate a scope — with no register access and no capability records; what a platform lacks shows
>   up as a scope with no indices (§3, §7.1).
> - **Three exceptions, not eight.** The five that only existed to carry client-side policy went with
>   the policy (§6).
>
> Sections not marked as revised are unchanged: §4.2's call-shape table is the plan for the core API
> at stages 3–7 and is unaffected, and §5, §7.2, §7.3, §8 and the appendices describe work later
> stages own.

## 1. Scope, and how to read it

This document fixes the **call shapes** of the `cryodaq` client interface — what a caller writes,
what comes back, what can go wrong — at the level a reviewer can judge the layer boundary and spot a
missing capability. It does not fix implementation: the walking skeleton that follows it is the
first code, and it will change details here. Where the document *proposes* rather than records, it
says so; the proposals are what review is for.

Four constraints bind every section. They are decided elsewhere and only restated here:

1. **Generality.** Channel geometry is reported by the platform, never assumed by the core; the
   tracking modality is a parameter; `connect()` does not presume a crate slot; nothing assumes
   configure-once-then-stream (proposal §Generality; SPECTRA's three sensor regimes).
2. **The operations criterion.** A procedure that sequences hardware reads and writes with decisions
   in between runs on the server as a named operation; the client supplies values, invokes by name,
   consumes the result. Analysis over data, plotting and orchestration are client-side, as is ad-hoc
   `get`/`set` (proposal §Decisions, *Operations*).
3. **Register paths live only in `cryodaq.platform`.** The client sees semantic names; the catalog
   is the firmware/software contract (proposal §Decisions, *Firmware/software boundary*).
4. **Compatibility.** The `SmurfControl` object sodetlib holds becomes a generated shim over this
   interface; every existing pysmurf attribute it has today — including those of `S.C` and `S.pub` —
   is preserved with today's semantics, and the shim fixes nothing (kick-off decisions 5 and 6).

**Reading the tables.** Every call-shape row names *today's* method with its definition line in
`python/pysmurf/client/` at `refactor/cryo-daq` `58ff4e41`, the **layer** it lands in
(`platform` · `core` · `application` · `utility`) and its **kind** under the criterion. The kind
vocabulary is the proposal's three values — `procedure` (server-side, registered, invoked by name),
`analysis` (client-side, over data), `capability` (something the platform reports or provides, which
the core may not assume) — plus three the classification needed to be total: `access` (a single
semantic-name get or set), `state` (an attribute of the session or its configuration) and `utility`
(logging, paths, publishing). The criterion classifies procedures; a bare register read is not a
procedure and a directory path is not analysis, and without the extra values a third of the compat
surface would have had no cell. Accepted 2026-09-10 (Tristan); the proposal's boundary check 2 is
amended to name the six. Appendix A also marks each name **µMUX** or **general**, the
generality axis.

## 2. Session

### 2.1 Construction and target — *revised 2026-09-11*

```python
def connect(target, *, timeout=30.0, monitor=True) -> Session: ...
```

`target` is an **endpoint**, in one of two forms: `"host:port"`, or `"crate:<slot>"` as shorthand for
the `9000 + 3·slot` port convention on this host (sodetlib `det_config.py:679`; server
`core/server_scripts/Common.py:109`), which appears nowhere else. There is no `"emulate"` form and no
`Target` record. An emulated system is a rogue tree with a ZMQ server on a port, so it is reached by
the same string as a crate and exercises the same code — which is the point: the one route in is the
one that talks to hardware. `connect()` **takes no configuration file** —
`SmurfControl.__init__`'s `ValueError('Must provide config file.')` (`base/smurf_control.py:95`) is
the behaviour being removed.

Which platform the server belongs to is decided by the **firmware loaded on the FPGA** — *revised
2026-09-14*: each map lists the firmware image names it covers, `connect` reads the build stamp
(`AmcCarrierCore.AxiVersion.BuildStamp`) and takes the image name from it, and firmware no map claims
is refused quoting what it read. Never by a start-up flag —
`SmurfApplication.StartupArguments` is `' '.join(sys.argv[1:])` (`core/_SmurfApplication.py:53`) and
says whatever the operator typed.

A tree with no firmware behind it reports an empty stamp — a register emulation reads as zeros — and
is refused too. Its platform is named instead, `connect(..., platform_name=...)`, and *revised
2026-09-15* that is **not a way past identification but a different question**: naming a platform is a
lookup in a table this package carries, so `connect` answers it with the rest of its arguments, before
it needs a server or rogue at all, and identification is never reached. `platform.identify` therefore
discovers and only discovers — it takes a tree and reads the stamp, with no argument that can overrule
what the hardware says. The consequence is stated where the argument is documented: a name is taken as
it stands, so a wrong one gives the wrong register map, and that shows up as names which do not resolve
rather than as a wrong value. `connect` logs the override for the same reason.

### 2.2 Connect versus reattach — *revised 2026-09-11*

Reattaching is the normal case and needs no flag: `connect` reads what the server says about itself —
`SmurfApplication.{SystemConfigured, ConfiguringInProgress, EnabledBays, StartupArguments,
SmurfVersion, JesdStatus}`, `AMCc.Ready` and the firmware identity — and returns it as
`sess.description`, a plain mapping. `sess.witness()` reads the stage-1 witness registers on request.
Nothing is inferred and nothing is refused: a caller that cares whether the system has been set up
reads `sess.description['configured']` and decides. Kick-off decision 2's `reattached` flag and the
`NotConfigured` refusal are dropped — a client that refuses work on a system it has only read one
register of is guessing, and the server's own dispatchers are what enforce order. The sidecar,
witness re-validation and raise-on-mismatch remain stage 4.

The live validation made the case concretely rather than in principle: the running server it attached
to reports `SystemConfigured` false, because nothing has set it up. A client that treated the flag as
a precondition would have been unusable against the only server there was — and unusable, in
particular, for setting it up.

### 2.3 Lifetime and transport policy — *revised 2026-09-17*

pyrogue is a dependency of `connect()` and of nothing else here, and the session it returns **is** a
rogue client: it holds a `VirtualClient` and `sess.root` is the server's tree as rogue presents it,
whole and unwrapped. The import sits at that one call site, so the package, the platform maps, the
endpoint arithmetic and the capture paths work on a machine that has no rogue — which is where the maps
get reviewed and where CI runs — while everything a session does needs the rogue the smurf image
brings. Nothing here comes from a package index by that name. Anything rogue can do with a node — a
variable's units, a process's `Progress`, a device's children, a YAML dump — is available without going
through this interface at all. What the interface adds over `root.getNode()` is a name that means the
same thing on every platform, and it adds nothing else.

The session is a **context manager** (pysmurf #1005): `with cryodaq.connect(t) as sess:` closes the
transport on exit and `sess.close()` does the same explicitly. The primitive exists — `VirtualClient`
has `__enter__`/`__exit__`, and `stop()` clears its `(addr, port)` singleton (`_Virtual.py:750-780`).
Transport policy is a **connect argument, not a global** (pysmurf #990, sodetlib #328): the link monitor
is left running where today's client turns it off (`_monEnable = False`), and `connect(timeout=…)` is
the bound on **one request** — today's fixed warn-5 s / fail-30 s pair (`base/base_class.py:106-109`)
becomes the default of an argument, which is what #990 asks for. It is set on the client when the client
is made, as today, and `timeout=None` is the way to ask for rogue's own behaviour of waiting
indefinitely. Stall reporting is rogue's, not re-wrapped here (§6). Long server work — a Process,
`setup` — is awaited with a polled wait on the process's own `Running` flag (§4.1), never a blocking
call with a single timeout, which is the failure mode of pysmurf #1015: the request bound and the
operation bound are different numbers, and a tuning run is limited by neither.

One limit to state, and it is the sharpest edge in this section: **concurrent sessions on one endpoint
in one process are not supported.** rogue caches one client per `(addr, port)` and `stop()` tears it
down (`_Virtual.py:750-780`), so two sessions opened on the same server in one process do not get a
socket each — they get one, and with it one lifetime and one transport policy. The first `close()` ends
both, and the last `connect()` sets the deadline and the monitor for both: a session opened with 30 s
and a session opened with 1 s cannot coexist on one endpoint, whatever the argument says. This is
rogue's semantics, and the alternative is bookkeeping of our own — a lock, a policy registry, endpoint
refcounting — which would be a second, partial cache beside rogue's, out of step with any other client
in the same process (a `SmurfControl` sets its own timeout on that same singleton and would be invisible
to it). So the limit is documented in `connect()` and stated here, and nothing pretends otherwise: a
program that needs two independent sessions on one server needs two processes. The usual case is one
session in one process, where none of this can arise; the case to keep in mind is an agent holding
clients in threads — `socs`'s `pysmurf_controller`.

`monitor=True` is therefore not a promise that a monitor is running, only that this session does not
stop one. rogue's monitor loop exits when the flag goes down and nothing starts it again
(`_Virtual.py:611`), so on a client that a `SmurfControl` in the same process has already disabled there
is no thread left to re-enable.

**Leaving the monitor running is a promise to end it.** rogue starts that loop in a thread that is not a
daemon (`_Virtual.py:512-514`), and CPython waits for such threads before it exits — so a session nobody
closed does not merely leak a socket, it holds the interpreter open for good. Measured on this tree: a
child that connects and returns without closing never exits, and the same child under `close()` is gone
in two seconds. Today's client avoids this by turning the monitor off at connect and keeping an
`atexit` handler for the socket (`base/base_class.py:106-111`); the handler is not what saves it, and
could not be — **the wait for non-daemon threads happens before `atexit` runs**, so an `atexit` hook is
already too late to stop the thread being waited for. Measured: registering `client.stop` with `atexit`
leaves the hang exactly as it was. What today's client actually relies on is the eager `_monEnable =
False`, which is to say it pays for a clean exit with the dead-server detection this section is keeping.
So `connect()` registers the close with `threading._register_atexit`, which runs while stopping the
client still means something. That name is CPython-internal — the stdlib's own `concurrent.futures` uses
it for the same reason — so it is looked up rather than assumed, and a Python without it gets a warning
and no monitor rather than a hang. The durable fix belongs upstream in one word (`daemon=True` on that
thread), and is §10's to carry; a background poller has no business outliving the program.

**Provisional shape.** One session per system. The transport is kept separable inside the session
so that a connection object holding several systems can be added later without changing any
operation signature. This is the proposal's deferred decision (§Decisions, *Interface shape*) and
is re-posed in §10.

### 2.4 Attributes — *revised 2026-09-11*

| Attribute | Shape | Today | Note |
|---|---|---|---|
| `sess.root` | the rogue tree | `S._root` via `VirtualClient` (`base/base_class.py:107`) | the whole server, unwrapped; the expert route and the escape hatch |
| `sess.pmap` | `PlatformMap` (§7.1) | `is_rfsoc`, `bays`, `crate_id`, `slot_number` set in `initialize()` (`smurf_control.py:255-262`) | which map this tree was recognised as; data, no hardware access |
| `sess.description` | mapping | `SmurfConfig` (`base/smurf_config.py`) held client-side; `SystemConfigured` server-side only (`roots/Common.py:379,404`) | what the server says it is, read once at connect; sidecar is stage 4 |
| `sess.get(name, *, index=-1)` / `sess.set(name, value, *, index=-1)` | semantic names (§3) | `_caget` / `_caput` (`command/smurf_command.py:141,49`) over register paths | a map lookup then `getNode(path).get()`; `index` is rogue's own, and is how one channel of a per-band array is reached. `set` is verified and blocking, rogue's own defaults, and **refuses a name the tree declares read-only** — nothing below refuses it, and where the value is the server's own the write would land |
| `sess.node(name)` | the rogue node | `S._root.getNode(path)` at each call site | for everything rogue offers that a name cannot express |
| `sess.call(name, *args, wait=None)` | the command's return value, or the process node | one method per operation, ~740 of them | §4.1 |
| `sess.validate()` / `sess.indices(scope)` | which of the map's names reach this tree, and why the rest do not; the indices of a scope | none — discovery is reading the source | §3.2 |
| `sess.witness()` | mapping | the stage-1 witness reads, by hand | §2.2 |
| `sess.paths` | `data`, `output`, `tune`, `plot`, `status` | `data_dir`, `output_dir`, `tune_dir`, `plot_dir`, `status_dir`, `base_dir`, `date`, `name` (`smurf_control.py:185-249`) | client-side bookkeeping; not sent to the server |
| `sess.log` | logger with `LOG_USER`/`LOG_INFO`/`LOG_ERROR` levels | `SmurfLogger` (`base/logger.py`), constants `base_class.py:64-72` | unchanged in kind |
| `sess.pub` | publisher with `register_file` and `publish` | `util/pub.py:61`, constructed `base_class.py:126` | injected; a null publisher when none is given |

`freq_resp`, `tune_file` and the `SmurfConfigPropertiesMixin` wiring values are **not** session
attributes: the first two are the tune record (`last_tune(sess)`, §4.3), the last are application
configuration reached through the shim's properties (§5, §8).

## 3. Names

### 3.1 Syntax — *revised 2026-09-11*, a starter set

Semantic names are dotted paths over **indexed scopes**: `band[b]`, `band[b].channel[c]`,
`bay[y]`, `bias_line[l]`, and unindexed `stream`, `timing`, `flux_ramp`, `firmware`. Examples, with
today's register-path template each replaces (`base/base_class.py:132-224`):

| Semantic name | Today | Kind of node |
|---|---|---|
| `band[b].tone.amplitude` | `{band_root}.CryoChannels.amplitudeScale` (`get/set_amplitude_scale_array`) | array variable |
| `band[b].tone.frequency` | `…CryoChannels.centerFrequencyMHz` (`get/set_center_frequency_array`) | array variable |
| `band[b].tone.frequency_offset` | `…toneFrequencyOffsetMHz` (`get_tone_frequency_offset_mhz`) | array variable |
| `band[b].feedback.enable` | `…feedbackEnable`, `…feedbackEnableArray` | variable, array |
| `band[b].eta.mag`, `band[b].eta.phase` | `…etaMag`, `…etaPhase` (`get/set_eta_mag_array`, `…_phase_array`) | array variables |
| `band[b].delay_us` | `…bandDelayUs` — a stage-1 witness | variable |
| `band[b].ops.gradient_descent`, `band[b].ops.start_gradient_descent` | `…CryoChannels.SerialGradientDescent` — a stage-1 Process — and `…runSerialGradientDescent`, the command that starts it, which is what a caller runs today | Process, command |
| `stream.downsample.factor`, `stream.filter.disable` | `SmurfProcessor.Downsampler.InternalFactor`, `…Filter.Disable` | variables |
| `flux_ramp.rate_khz`, `flux_ramp.fraction_full_scale` | `RtmCryoDet.RampMaxCnt`, `…LTC1668RawDacData` | modality-scoped |
| `firmware.build_stamp`, `firmware.git_hash` | `AxiVersion.BuildStamp`, `…GitHash` | platform-scoped |

Rules: indices are the caller's, and the tree reports which ones exist (`sess.indices('band')`).
**A per-channel register is one array, and a channel is `index=`** — rogue's own argument, so
`sess.get('band[4].tone.amplitude', index=17)` reads one channel and there is no `channel[*]` scope.
Names are stable across firmware releases; paths are not.

Two kinds of name the draft proposed are **dropped**. A name **derived** rather than stored —
`band[b].channel[c].frequency_mhz`, today `channel_to_freq` (`util/smurf_util.py:2521`), or
`band[b].channels_on`, today `which_on` (`:2110`) — is a computation over registers, not a register:
putting it in the map made `get()` two resolution paths with a callback in one of them, for two
functions that are three lines of arithmetic over names the map already has. They belong in whatever
layer wants them, as functions of a session. And a name is not gated on a declared capability
(§7.1): what a tree does not have shows up as a scope with no indices, so the name is never offered
rather than being withheld by a rule.

**Sparingly.** The map is a name → path table, and the client today already has one: 216
register-name constants in `command/smurf_command.py`, with band-index arithmetic at every call site
and nothing checking them against firmware. That is the duplication the proposal removes, and a map
re-grows it unless it is kept small on purpose. Rule: **a name enters the map only when a core
operation or the generated shim needs it.** One indexed entry covers every band; the map lives in
`cryodaq.platform` and nowhere else; and CI is to validate every entry against both released firmware
packages, so an entry that drifts fails a build rather than a night on the crate — the gate is stage
3's, and §10 says why it cannot be built yet. Everything not in
the map is reached through `sess.root` — the tree itself, which is also what the shim's
`_caget`/`_caput` forwarders need. The table above is a starter set, to be pared or extended at
stage 3.

### 3.2 Resolution and validation — *revised 2026-09-11*

Resolution is **a dictionary lookup and a string format**: the name's pattern indexes the platform
map, which yields a path template and the kind of node it is; the indices fill the template;
`root.getNode(path)` returns the node, loading only the devices on that path — a few round trips per
band, fully usable nodes, and no change to rogue. There is no catalog object, no entry record and no
resolver protocol. That is kick-off decision 1 **narrowed twice**: on 2026-09-10 its second half — a
one-round-trip enumeration through an attribute rogue does not expose — was dropped and no rogue issue
filed; on 2026-09-11 the client-side object model went too.

`sess.validate()` is an expert tool — it expands each pattern over the indices the tree turns out to
have, one `getNode` per name, a few hundred round trips, never on the normal path — and reports each
failure with the name, the path tried and why. There is deliberately **no** method that lists names
without checking them: what the map offers a platform and what a particular build of that platform's
firmware actually carries are not the same set, and a list that did not say which it was would invite
callers to iterate names that cannot resolve. Whether the map matches released firmware is a **CI job**
over the two pyrogue packages — the intended gate, deferred to stage 3 (§10). Resolution failure raises
`UnresolvedName`; it never returns `None`.

Stage 2 fixes the syntax and this contract; the map's *content* is stage 3's. The 20 `CONTRACT`
names of `scripts/stage1_client_check.py` are its first entries.

## 4. Operations

### 4.1 Invocation shape — *revised 2026-09-11*

**An operation is a rogue node, and it is handed to the caller as one.**

```python
sess.call("band[4].ops.load_tune_file")                    # a command: called, its value returned
sess.set("band[4].ops.gradient_descent.max_iters", 15)     # parameters are registers
proc = sess.call("band[4].ops.gradient_descent", wait=30)  # started, waited on, node returned
proc.Message.get(); proc.Progress.get(); proc.Running.get()
sess.stop("band[4].ops.gradient_descent")                  # its own Stop
```

`call` is the whole of it. For a command it calls the node (`_Virtual.py:97-98` makes both kinds
callable through `VirtualClient`) and returns what the command returned. For a process it calls the
process's own `Start` and, with `wait=<seconds>`, polls the server's own `Running` until it clears or
the bound expires — `TimeoutError` then, with the process untouched and pollable. It returns the
process node. `wait` bounds the process; each read of `Running` is bounded separately by the session's
`timeout` (§2.1), so a link that stops answering ends the wait as a transport failure rather than as a
`TimeoutError`, and only a session opened with `timeout=None` can wait on one indefinitely. It invents no verdict: what happened is in `Message`, which is the server's word for it.

Both bounds are judged **before `Start`**, with `ValueError`: a bound that makes no sense should not be
able to leave a process running behind a call that then failed on its way to waiting for it. `wait=0`
reads the flag once and `wait=math.inf` waits as long as the process takes — a wait never leaves Python,
unlike the request deadline of §2.1, which has to reach rogue as an integer number of milliseconds and so
cannot be infinite. A negative wait is refused, and so is a `poll` that is not finite and positive: zero
would spin on the transport rather than sleep between reads.

Dropped from revision 1, with the reason: `OperationSpec` and `Provider` (a declaration layer over
nodes rogue already describes — `node.description` is the documentation, and what exists is what the
tree has); `Param` (typed coercion in front of registers that have types); `Result` (a record built
from reads the caller can do); `Handle` (`status`/`progress`/`message`/`wait`/`cancel`, each a thin
forward to a child node of the process). Two known bounds come with handing the process over rather
than judging it, and both are honest about what the server offers:

- **A short run is indistinguishable from no run.** A wait ends when `Running` is false, which is also
  true before the process starts. On the emulated trees both tuning processes finish inside one poll
  interval. Revision 1 answered this with a two-second start grace and a scan of the message text for
  the words *error* and *failed*; that is a client inventing a verdict rogue does not report, so it is
  gone. `Message` is the signal, and rogue writing a status code is the fix.
- **Which gradient descent ran is a register, so the map names it.** The dispatcher
  `runSerialGradientDescent` reads `UseNewSerialGradientDescent` and starts one of two processes. The
  map therefore names both processes *and* the selector (`band[b].ops.gradient_descent`,
  `…new_gradient_descent`, `…use_new_gradient_descent`), so a caller can see which one it is watching
  instead of assuming. Revision 1 named only the first and would have reported success for a process
  that never ran.

### 4.2 Call shapes

One row per draft call. *Today* cites the method the row replaces; where several methods collapse
into one call, the principal one is cited and the rest are in Appendix A. This table is the plan for
the **core API**, stages 3–7; the 2026-09-11 revision does not touch it, except that rows whose kind
is `access` are sugar over `sess.get`/`sess.set` and rows whose kind is `procedure` are reached by
`sess.call` rather than `invoke`. `set_amplitude_scales` is no longer among them: it is a
register-specific setter, and `sess.set("band[b].tone.amplitude", v)` is the semantic form. It stays
in the map as a command only because the compat surface reaches the dispatcher.

| Call | Today | Layer | Kind | Note |
|---|---|---|---|---|
| **bring-up** | | | | |
| `setup(sess, cfg) -> Result` | `setup` `base/smurf_control.py:322` | core | procedure | server owns the procedure; the client supplies `cfg`. Today's 6 `is_rfsoc` branches inside it become platform dispatch |
| `check_ready(sess) -> report` | `check_jesd` `util/smurf_util.py:2322`; `get_jesd_status` `command/smurf_command.py:5519`; `check_adc_saturation` `util/smurf_util.py:1717` | platform | procedure | witness registers, JESD, timing; the only operation besides `setup` allowed on a `NotConfigured` session |
| **tones** | | | | |
| `set_tone_amplitudes(sess, band, amps)` | `set_amplitude_scale_array` `command/smurf_command.py:1873` | core | access | sugar over `sess.set("band[b].tone.amplitude", …)` |
| `set_tone_frequencies(sess, band, offsets)` | `set_center_frequency_array` `command/smurf_command.py:3205` | core | access | as above |
| `set_attenuation(sess, band, uc=, dc=)` | `set_att_uc` `command/smurf_command.py:4883`; `set_att_dc` `:4956` | platform | capability | **proposed move**: the draft has this under core tones, but RFSoC eval boards have no attenuators — an optional capability, `platform.attenuators` |
| `enable_feedback(sess, band, mask)` | `set_feedback_enable` `command/smurf_command.py:3086`; `…_array` `:1950` | core | access | |
| `band_off(sess, band)` / `channel_off(sess, band, ch)` | `band_off` `util/smurf_util.py:2180`; `channel_off` `:2194`; `all_off` `:3805` | core | procedure | several writes, no decisions — registered so that "off" is atomic |
| **resonators** | | | | |
| `full_band_response(sess, band) -> sweep` | `full_band_resp` `tune/smurf_tune.py:583` | core | procedure | raw-IQ capture and transform; the sweep is data |
| `find_freq(sess, band) -> sweep` | `find_freq` `tune/smurf_tune.py:3473`; `full_band_ampl_sweep` `:3665` | core | procedure | |
| `assign_channels(sess, band, freqs) -> assignment` | `assign_channels` `tune/smurf_tune.py:1640` | core | procedure | needs geometry from the platform, so not client analysis |
| `setup_notches(sess, band, assignment)` | `setup_notches` `tune/smurf_tune.py:3892` | core | procedure | |
| `eta_scan(sess, band, channels) -> eta` | `run_serial_eta_scan` `command/smurf_command.py:1135`; `eta_scan` `tune/smurf_tune.py:2238` | core | procedure | stage-1 Process `SerialEtaScan` |
| `gradient_descent(sess, band)` | `run_serial_gradient_descent` `command/smurf_command.py:1161` | core | procedure | stage-1 Process; its parameters `set_gradient_descent_*` `:744,910,964` are `access` on `band[b].ops.gradient_descent.*` |
| `estimate_phase_delay(sess, band)` | `estimate_phase_delay` `util/smurf_util.py:328` | core | procedure | named in the operations decision; missing from the draft block |
| **tracking** | | | | |
| `tracking_setup(sess, band, modality, …) -> result` | `tracking_setup` `tune/smurf_tune.py:2494` | core | procedure | today's is flux-ramp specific (`reset_rate_khz`, `fraction_full_scale`); the modality argument is where that goes (§7.2) |
| `relock(sess, band)` | `relock` `tune/smurf_tune.py:1912` | core | procedure | |
| `estimate_lms_freq(sess, band)` / `optimize_lms_delay(sess, band)` | `estimate_lms_freq` `tune/smurf_tune.py:4505`; `optimize_lms_delay` `:4361` | core | procedure | flux-ramp modality operations |
| `check_lock(sess, band) -> report` | `check_lock` `tune/smurf_tune.py:3315` | core | procedure | |
| `flux_ramp_setup(sess, rate_khz, fraction)` / `flux_ramp_off(sess)` | `flux_ramp_setup` `tune/smurf_tune.py:3173`; `flux_ramp_off` `command/smurf_command.py:8034` | core | procedure | **belongs to the flux-ramp modality**, not the core API proper; registered by the modality |
| `set_fixed_flux_ramp_bias(sess, frac)` / `unset_…` | `set_fixed_flux_ramp_bias` `tune/smurf_tune.py:3108`; `unset_…` `:3081` | core | procedure | declared gap; same home as `flux_ramp_setup` |
| **data** | | | | |
| `with stream(sess, *, rate=, format=, channels=) as s:` | `stream_data_on` `util/smurf_util.py:984`; `stream_data_off` `:1160`; `take_stream_data` `:865`; `set_stream_enable` `command/smurf_command.py:2181` | core | procedure | `s.path`, `s.rate`, `s.format` report what is actually coming out, per channel; `take_stream_data` is the context plus a timed wait |
| `set_downsample(sess, factor)` / `set_filter(sess, coeffs)` | `set_downsample_factor` `command/smurf_command.py:10219`; `set_downsample_filter` `util/smurf_util.py:3451`; `set_filter_disable` `:10324` | core | access | coefficient design is client analysis; the write is access |
| `read_stream(path) -> timestreams` | `read_stream_data` `util/smurf_util.py:1183` | core | analysis | client-side, no hardware |
| `capture_raw(sess, band, nsamp) -> iq` | `read_adc_data` `util/smurf_util.py:1773`; `take_debug_data` `:43` | core | procedure | the raw-IQ modality's primitive; also what `full_band_response` is built on |
| **tune records** | | | | |
| `save_tune(sess) -> path` / `load_tune(sess, path)` / `last_tune(sess)` | `save_tune` `tune/smurf_tune.py:4273`; `load_tune` `:4289`; `tune_file` `:4283`; `freq_resp` `base/base_class.py:240` | core | procedure | the record becomes server-authoritative; `load_tune` writes it to hardware, so it is a procedure by the criterion |
| **analysis** (client) | | | | |
| `find_resonators(sweep) -> freqs` | `find_peak` `tune/smurf_tune.py:804` | core | analysis | |
| `fit_eta(eta)` / `summarise_tracking(result)` | `eta_fit` `tune/smurf_tune.py:1229`; `analyze_psd` `debug/smurf_noise.py:1434` | core | analysis | |
| `plot_*(…)` | `plot_tune_summary` `tune/smurf_tune.py:437` and the `make_plot=` branches throughout | core | analysis | plotting leaves every procedure; procedures return data |
| **application-registered** (pysmurf registers; the core never mentions) | | | | |
| `overbias_tes(sess, bias_groups, …)` | `overbias_tes_all` `util/smurf_util.py:3080`; `overbias_tes` `:3019` | application | procedure | `Operation` over `platform.bias_lines` |
| `take_iv(sess, bias_groups, …) -> iv` | `run_iv` `debug/smurf_iv.py:30` | application | procedure | the sweep is the operation; `analyze_iv` `:591` stays client analysis |
| `bias_step(sess, …)` | `bias_bump` `util/smurf_util.py:3576` | application | procedure | |
| `set_current_mode(sess, bias_groups, mode)` | `set_tes_bias_high_current` `util/smurf_util.py:3158`; `…_low_current` `:3209`; sodetlib's bypass `util.py:601` | application | procedure | the public home sodetlib #482 asks for: relay flip and DAC rescale as one atomic server operation |
| `play_waveform(sess, bias_lines, table)` | `play_sine_tes` `util/smurf_util.py:3912`; `set_rtm_arb_waveform_*` `command/smurf_command.py:6241-6547` | application | procedure | over a **proposed** optional capability `platform.bias_lines.waveform`; declared gap |
| **no public home** — shim only | | | | |
| `get_timestamp()` | `get_timestamp` `base/smurf_control.py:740` | utility | utility | client wall clock; declared gap, proposed: not part of the interface |
| `_caget(path)` / `_caput(path, v)` / `_cryo_root(band)` | `command/smurf_command.py:141,49`; `base/base_class.py:327` | core / platform | access | path-string access does not survive as public API; the shim keeps it forwarding (decision 6) |

The rows marked *proposed* — attenuators as a capability, the waveform capability, `get_timestamp`
dropped, the flux-ramp group homed in the modality — are the design questions this table surfaces;
they reappear in §10.

**Fidelity first, then review.** The first implementation of each row does what today's method does,
moved to where the criterion puts it, and nothing more — a re-organisation, which is what makes the
compat contract checkable row by row. Reviewing these procedures on their merits (redundant reads,
fixed sleeps, plotting entangled with measurement, parameters that should be configuration) is a
**named later task**, opened once the skeleton passes the spine, not folded into the move.

### 4.3 What an operation may assume

Nothing about geometry (it reads it: `band[b].n_channels` and its four siblings, §3.1), nothing about
bias hardware except that the names it uses resolve, and nothing about the data path being in a steady
streaming state. The last is the timed-pulse constraint: an operation that captures, changes the
hardware and captures again must be expressible, which rules out any API that starts a stream at
connect and assumes it runs.

## 5. Config objects

One rule, from the proposal (§Configuration): **only operations that change the configuration
take a config argument.** `setup(sess, cfg)` does. `tune_band(sess, band)` does not — it reads what
it needs from `sess.description`. This removes most of what `base/smurf_config_properties.py`
(2,597 lines, 63 properties with setters) exists to do, which is to carry configuration values
alongside the object so every method can reach them.

At stage 2 `cfg` is an **opaque mapping with a `hash`**; schema, inheritance and the shipped default
are stage 4 and change nothing in the shapes here. *Revised 2026-09-11:* `sess.description` is a plain
mapping of what the server says about itself, read at connect — the endpoint and time it was read, the
root's name, `SystemConfigured`, `ConfiguringInProgress`, `Ready`, `EnabledBays`, `StartupArguments`,
`SmurfVersion`, `JesdStatus` and the firmware version, build stamp and git hash. The witness registers
are `sess.witness()`, read when asked rather than at every connect; `config`, `hash` and the published
resolved values arrive with `setup` at stage 4, and `reattached` is gone (§2.2). The
wiring and calibration constants — `R_sh`, `pA_per_phi0`, `bias_line_resistance`,
`pic_to_bias_group`, `bias_group_to_pair`, `high_low_current_ratio` — are **application**
configuration: `cryodaq` never reads them; pysmurf's registered operations do, and the shim's
properties forward to them.

## 6. Error model — *revised 2026-09-11*

There is no error model to inherit. The client today raises only builtins (`ValueError` ×40,
`RuntimeError`, `TimeoutError`, `ConnectionError`), returns `None` from `_caget` when offline or
`execute=False` (`command/smurf_command.py:186-189`) and returns sentinels from the cryocard —
`read_relays` gives `80000` on failure (`command/cryo_card.py:160-166`), `get_fw_version` gives
`(0, 0, 0)` for "no card" (`:347`). Three conventions, none documented. The proposal fixes only the
stage-4 case: disagreement between server and sidecar raises naming the field and both values.

**Three exceptions**, one base so callers can catch the interface as a whole. There are three because
there are three things this layer can itself be wrong about: the connection, the name, and a bounded
wait that ran out.

| Exception | Raised when | Carries |
|---|---|---|
| `CryodaqError` | base class | — |
| `ConnectError` | the endpoint does not parse, no server answers it, or its tree matches no platform map | the endpoint and what was tried |
| `UnresolvedName` | a name is not in the map, its register is not in this tree, or it is used as the wrong kind | the name, the pattern or path tried, the reason |
| `TimeoutError` (builtin) | a process is still running when `call(..., wait=…)` expires | the name and the bound |

The five in revision 1's table are gone, each with its cause:

| Was | Why it went |
|---|---|
| `NotConfigured` | the client-side refusal it served is dropped (§2.2); `sess.description['configured']` is a read and the caller decides |
| `CapabilityMissing` | there are no capability records to be missing (§7.1); an absent name raises `UnresolvedName` naming the path |
| `TransportError` | a stall is rogue's own report and re-wrapping it added a translation layer with no new information |
| `OperationError` | the client no longer judges whether a process failed; `Message` is the server's word (§4.1) |
| `DescriptionMismatch` | declared for stage 4 and raised by nothing; it arrives with the sidecar that needs it |

Rules: a failed lookup **raises**; the interface never returns `None` for "could not read" (the *shim*
keeps today's `None`-when-offline because decision 6 says it preserves current usage). No sentinels.
The rule is about a read the caller asked for **by name**: `get(name)` on a name this tree does not have
raises, and never answers `None`. Two places carry a `None` and are not exceptions to it but a different
question — `Session.description`'s optional fields and `witness()`'s per-name entries are *surveys*,
asked over a list of names that a given tree may or may not have, where "this tree does not offer it" is
the datum being collected and the alternative would be a survey that raises on its first absence. Both
are read-only records, and neither is a value a caller asked for; that distinction is the rule, and it is
stated here because a reader checking the rule against `_optional_get` will otherwise find a
contradiction that is not one.
Retries are transport policy set on the session, never written into individual operations — the
cryocard's hand-rolled 5× loop is the pattern being retired. Bound stated honestly: rogue flattens
every transport failure to bare `Exception` (`_Virtual.py:718`, `_ZmqServer.py:92-96`), so a caller who
needs to tell a dead link from a refused write reads rogue's message; narrowing that is rogue's to do,
and inventing our own two names for it here would only have hidden that it is not narrowed.

## 7. The three seams

### 7.1 Platform capability set — *revised 2026-09-11*

**A platform is a map, and what it lacks is a name that is not offered.** `cryodaq.platform` holds one
module of data per generation of hardware — register path templates, the name table built from them,
the witness set, and how to enumerate each indexed scope — plus the lookup over it. It reads no
register and imports no rogue; a boundary check enforces both. Nothing that a client does (get, set,
call, poll a process) lives there, because that would be the same work in two layers.

Capabilities are therefore **discovered, not declared**. A scope is enumerated by asking the tree which
indices it has: an RFSoC has no `AppTopJesd[*]` and no `MicrowaveMuxCore[*]`, so `sess.indices('bay')`
is empty there and no `bay[…]` name is even a candidate for `sess.validate()`. That is the same fact the
capability records carried, obtained from the tree instead of from a second description of it that
could disagree with it. Revision 1's typed records (`Geometry`, `Bays`, `Firmware`, `Transport`,
`Timing`, `Attenuators`, `DataLinks`, `BiasLines`, `Cryocard`, `Amplifiers`, `FluxRamp`), the
`requires=` gate on a name, and `CapabilityMissing` are all gone. Channel geometry, which was
`Geometry`'s reason to exist, is five registers and is now five names —
`band[b].{center_mhz, digitizer_rate_mhz, n_subbands, n_channels, channel_bandwidth_mhz}` — read when
wanted rather than at every connect.

The table below is kept as the **inventory of what a platform must be able to express**, which is what
reviewers were asked to judge; read the *Capability* column as a family of names and the *Required*
column as whether the core may assume the family resolves.

| Capability | Fields | Today (the getter each field is derived from) | Required |
|---|---|---|---|
| `geometry` | `n_bands`, `band_centers_mhz`, `digitizer_rate_mhz`, `n_subbands`, `subband_width_mhz`, `n_channels`, `channelizer_bin_width_mhz` | `get_band_center_mhz` `command/smurf_command.py:4303`; `get_digitizer_frequency_mhz` `:4373`; `get_number_sub_bands` `:504`; `get_number_channels` `:537`; `get_channel_frequency_mhz` `:4338`. The µMUX literals live only behind `get_subband_centers(hardcode=True)` `util/smurf_util.py:2660` | yes |
| `bays`, `boards` | populated bays, board inventory | `which_bays` `util/smurf_util.py:2437`; `EnabledBays` | yes |
| `firmware` | `build_stamp`, `git_hash`, `version` | `get_fpga_build_stamp` `:5657`; `get_fpga_git_hash_short` `:5636` | yes |
| `transport` | kind (PCIe / Ethernet / emulated), endpoint | the root class chosen in `core/roots/` | yes |
| `timing` | mode, external reference present | `setup()`'s timing branch | yes |
| `attenuators` | `uc`, `dc` per band | `set_att_uc`/`_dc` `:4883,4956` | **optional** (proposed) |
| `bias_lines` | count, DAC bits, volts/bit, `waveform` sub-capability | `_rtm_slow_dac_nbits`/`_bit_to_volt` `base_class.py:246-249`; `set_rtm_slow_dac_volt_array` `:7475`; `set_rtm_arb_waveform_*` `:6241-6547` | optional |
| `cryocard` | revision, relays, AC/DC mode, temperature | `CryoCard` `command/cryo_card.py:36`; `get_cryo_card_relays` `:8657` | optional |
| `amplifiers` | inventory per card revision, gate/drain controls | `set_amplifier_bias` `:7992`; `C.list_of_c*_amps` `cryo_card.py:68-70` | optional |
| `flux_ramp` | generator present, rate range | `RtmCryoDet` nodes via `flux_ramp_setup` | optional — its absence is what makes a raw-IQ-only platform describable |

The prior art (`old_workspace/…/core/operations/_platforms.py`) had the booleans and no geometry;
geometry is the addition SPECTRA forces.

**Decided 2026-09-10 (Tristan):** attenuators and the RTM waveform are optional capabilities. With
the caveat that this must not become cumbersome as platform diversity grows — which is what the
2026-09-11 revision acts on. The rules that decision fixed all survive, in a cheaper form:

- *Never inferred from the platform's name or firmware string.* Amended 2026-09-14, and the amendment
  is narrower than it looks. What that rule was aimed at is branching on something the **caller**
  supplies — an `is_rfsoc` flag, a name in a configuration file — and that is still forbidden;
  `is_rfsoc` remains what the grep gate refuses. Identification now reads the build stamp the FPGA
  reports, which is not a string about the platform but a measurement of it, taken from the system in
  hand. The rule that survives is therefore *identity comes from the tree, never from the caller's
  configuration*, and the core still branches on nothing at all.
- *A new capability is not a new subclass or a new root.* Now it is not even a new record: it is
  entries in a map, and their absence on a tree that lacks the hardware needs no expression.
- *An operation depends on a capability or does not.* A name resolves or raises `UnresolvedName`
  naming the path that was not there. There is nowhere to put a partial-support flag.
- *A whole new generation of hardware is a second module in the map registry* — recognised by the
  firmware it runs, not by a branch inside the first. The ATCA carrier and the RFSoC are two such
  modules as of 2026-09-14, sharing one register table between them.

**The carrier and the RFSoC are two platforms sharing one register table** — *decided 2026-09-14, in
review*. Divergence between them is concentrated in **procedure**, not paths: `is_rfsoc` appears in 26
places in today's client and **7 of those are inside `setup()`** (proposal §Measured), so bring-up
ordering, what is configured and in what sequence are where the two genuinely part company. Their
register paths agree, and `platform/_atca.py` and `platform/_rfsoc.py` accordingly take their table
from `platform/_umux.py` and differ in the firmware each claims. What that buys before `setup()` moves
(stage 5) is the seam it will need: the two are told apart from the beginning, so the procedures land
against platforms that already exist rather than forcing a split at the moment they are written. The
rule that makes it safe is the one above — a branch on platform identity inside a module is the failure
mode, not the fix.

The RFSoC entry's **register table is validated, its firmware image names are not.** The RFSoC's own
firmware package is a subclass of the carrier's that defaults `isRFSOC` on and does nothing else, so its
tree can be built without that package and the emulated gate identifies `umux-rfsoc` on it with an empty
bay scope. The three image names the map claims, on the other hand, are derived from the build targets of
the two RFSoC firmware repositories rather than read off a board — no RFSoC is reachable from where this
work is done. Confirming them is stage 7's (`cryodaq.platform` and RFSoC parity), and until then an RFSoC
whose stamp says something else is refused by name rather than mis-identified. Confirmed as the plan on
2026-09-15 (Tristan): *"ok to delay nailing down the exact tags until we have a system to test with"* —
the map says so in place, so the names are read as a derivation awaiting a measurement, and `platform_name=`
names the platform meanwhile.

**Asked for 2026-09-11, then decided 2026-09-14 (Tristan)**, on reviewing the rebuilt structure:
*"they will diverge more when we get to implementing `setup()` and other operations. We should avoid
creating another situation where different platforms are not well separated."* And on the review of the
implementation: *"RFSoC and ATCA should be separate platforms: although the trees are very similar, the
omitted hardware registers are indicative that the hardware is different and as such will require a
different series of initialisation steps."*

### 7.2 Modality — *provisional*

A modality is **how a channel is tracked and at what rate and format it is emitted**. It is a
**discriminated configuration value**, not a class hierarchy: the three known values are `flux_ramp`
(today's µMUX tracking — the only one
the compat contract protects), `raw_iq` and `direct`. Each names the processor-chain stages it
enables (every `_SmurfProcessor.py` stage already has a `Disable`), the per-channel rate and format
it emits (`DownsamplerExternalBitmask` is already per-channel), and the operations it registers
(`flux_ramp_setup`, `flux_ramp_off`, `set_fixed_flux_ramp_bias`, `estimate_lms_freq`,
`optimize_lms_delay` belong to `flux_ramp` and are meaningless elsewhere). `tracking_setup` takes
the modality as an argument. A class is the natural implementation later; the interface commits
only to the discriminated value so the skeleton is not blocked on it.

**Nothing carries it yet, and the interface does not pretend to** — *stated 2026-09-15*. An earlier
draft of this section had the session carrying it; it does not. `sess.description` is what the server
says it is, read once at connect, and it has no modality entry: the value belongs to configuration
rather than to what a tree reports, and where it lives is the seam the processor-chain work opens. When
it arrives it is read as `sess.description['modality']` — a mapping key, since `description` is a
mapping and not an object.

This section is the least settled in the document and is expected to change as the skeleton meets
the processor chain. What is binding now is only the paragraph below — the list of what the API must
not assume. Everything above it is revisited at implementation.

What the API **must not assume**, so that QICK-style timed pulse sequencing is not precluded:
that a stream is running after `setup`; that all channels share one rate; that a flux ramp exists;
that tracking is continuous rather than triggered.

### 7.3 Operation registration — *deferred past stage 2, 2026-09-11*

This seam is **server-side** and stage 2 no longer opens it. Moving
`pysmurf.core.operations` into `cryodaq`, and with it any declaration of what a provider attaches,
waits until the platform layer exists to attach against (Tristan, 2026-09-11: *"if we are not ready to
implement platforms yet, this can wait"*). Until then the server stays the tree it is today — a rogue
tree with `attach_all_cryo_operations` wired into `Common.__init__` — and the client reads the nodes
that puts there. Read the rest of this section as the design for that later step, with three
corrections from the 2026-09-11 revision: the client no longer holds `OperationSpec` or `Provider`
(§4.1), so whatever the attach declares is for the *server's* benefit; the *Discovery* and
*Progress and status* rows are answered by the tree itself — the map's own entries, checked against it
by `sess.validate()`, and the process's own child nodes — rather than by a client-side catalog view or
handle; and the *Requires* row's `requires=` gate and the `CapabilityMissing` it raises are gone with
the capability records (§7.1, §6), so what a provider may assume is that the names it uses resolve.
`sess.operations` in the *Discovery* row is the same case: it is what that later step would add, and
no session has it today.

A **provider** is what the server composition attaches; the µMUX modality of `cryodaq` is one,
`pysmurf` (the TES operations) is another. Stage 1 left exactly one, hard-wired: `Common.__init__`
calls `attach_all_cryo_operations(self._fpga)` unconditionally between `self.add(self._fpga)` and
`start()` (`core/roots/Common.py:76`), and that function adds the same 29 nodes to every
`Base[i].CryoChannels` it finds — 19 `LocalVariable` parameters, 4 `pyrogue.Process` algorithms, 6
`LocalCommand` dispatchers (`_CryoOperations.py:81-121`). The protocol below is what that module
would have to *declare* to be one of several; each row names the stage-1 fact it generalises.

```python
class Provider(Protocol):
    name: str                                   # "cryodaq.umux", "pysmurf.tes"
    def operations(self, platform: Platform) -> list[OperationSpec]: ...

OperationSpec(name, scope="band" | "system", inputs=(Param(name, type, default, doc)…),
              result=…, requires=(capabilities…), doc=…)
```

| Declares | Stage 1 today | Provider protocol |
|---|---|---|
| **Identity** | the `label=f'Base[{i}].CryoChannels'` string in the log and refuse message (`:146-150`) | `Provider.name`; appears in the refuse message and in every catalog entry it creates |
| **Operations** | 4 Processes + the 6 commands that start or wrap them; which is which is convention | one `OperationSpec` per operation. A `Process` is the *only* long-running shape (§4.1); the 6 dispatchers are what `invoke` does and are not declared separately |
| **Inputs** | 19 RW `LocalVariable`s the client sets *before* running (`gradientDescentStepHz`, `etaScanDelF`, …), read by the Process off its parent (`_SerialEtaScan.py:49-53`) | `Param`s with typed defaults. Each remains a per-scope parameter node, so `invoke(name, step_hz=…)` and the shim's set-then-run (`set_gradient_descent_step_hz` → `run_serial_gradient_descent`) both work |
| **Requires** | implicit: `SerialFindFreq(ch._n_channels, ch._freqSpanMHz)` reads geometry from the device (`:374`) | `requires=` names §7.1 capabilities; `geometry` for all four, `tones`, `feedback` as used. Attach refuses with `CapabilityMissing` when one is absent — the core assumes nothing |
| **Where it attaches** | the module walks `fpga.AppTop.AppCore.SysgenCryo.Base[i].CryoChannels` itself (`:189`) — a register path outside the platform layer | the platform hands the provider its scope devices (`platform.scopes("band") -> [(i, device)]`); the provider never names a path. This is the one boundary fix stage 2 owes the stage-1 code |
| **Refuse** | atomic: any one of the 29 names already present raises before anything is added; `_exists` is pyrogue's own `add()` collision predicate (`:124-166`) | same predicate, same atomicity, per provider; plus the capability refuse above. Refusal is a `RuntimeError` at composition, never a runtime surprise |
| **Progress and status** | `Process.Running / Progress / Message / Step / TotalSteps` (`_Process.py:64-109`), plus the per-band mutex `etaScanInProgress`, set 1→0 by *all four* Processes and by `runEtaScan` (`_SerialEtaScan.py:42`, `_SerialFindFreq.py:46,105`, `_CryoOperations.py:446-493`) | `h.status / h.progress` (§4.1) read the Process nodes; the mutex is one catalog name per scope, `band[b].ops.in_progress` — not per operation, because it is not one today |
| **Documentation** | `description=` on every variable and command, asserted non-empty by `validate_cryo_operations.check_descriptions_survive` | `doc` on every `OperationSpec` and `Param`; the catalog entry is generated from it; CI asserts it, as today |
| **Discovery** | none — the client knows the path | `sess.operations` lists the entries the attach created, with `doc`, `requires`, `scope`; a name is confirmed against the tree by the decision-1 route on first use |

Attach happens where stage 1 attaches — after the FPGA device exists, before `Root.start()` seals
the tree — and in composition order; `cryodaq` never imports an application, the composition (an
application entry point) imports both, and the import-direction test guards that. Every
`OperationSpec` becomes a rogue Process (or, for `scope="system"` one-shots, a command) under a
`Root`-visible path and a catalog entry under `band[b].ops.<name>` or `ops.<name>`. What the
protocol does **not** fix: the Python base class the four Processes share (they have none today —
each subclasses `pyrogue.Process` directly), and whether the 19 parameters stay on the Process's
parent or move under it; both are skeleton-level choices. `validate_cryo_operations.py`'s twelve
checks are the acceptance test of the first provider and carry over unchanged in intent.

## 8. Compatibility — the generated shim

Decisions 5 and 6 fix the contract; this section states what the generator therefore has to
produce. The shim is `pysmurf.client.SmurfControl`, constructed over a `cryodaq` session, and its
surface is **every existing pysmurf attribute** of `SmurfControl` and of its pysmurf-owned
sub-objects that sodetlib reaches. The index into that surface is the 156-name artifact
(`artifacts/stage2/sodetlib_S_attrs_tpm-rogue6_04cfa27.txt`, 17 private) plus its one-level
descent (`…_sub_attrs_…txt`: 9 `S.C.*`, 4 `S.pub.*`). `S._sodetlib_cfg` is sodetlib's own and is
outside the contract.

The forwarders: public methods forward to the named forms of §4.2 or to `sess.get`/`sess.set`;
`_caget`/`_caput` accept a register path and forward with today's `wait_done`/`count`/`index`
behaviour; `_cryo_root(band)` returns the resolved path prefix as a string; `_rtm_slow_dac_data_reg`
and the other `_*_reg` constants remain strings; the nine wiring properties forward to the
application configuration; `C` and `pub` remain the real `CryoCard` and `Publisher`. The shim
**fixes nothing** — sodetlib #482's bypass keeps working through it; the fix is the public
`set_current_mode` operation in §4.2, on its own schedule.

The **compat contract test** freezes this surface from stage 2 onward. It is derived from PR #798's
`tests/ci/check_so_interface.py` (pure `ast`, no dependencies) with the frozen dictionary
*generated from the two artifacts* rather than hand-listed, and its three unlisted names (`setup`,
`set_mode_dc`, `set_mode_ac`), dead `child` branch and source-string default comparison fixed. It
asserts existence and signature; deprecation notes are Appendix C, not failures.

## 9. Pass conditions this specification carries

From the proposal's boundary checks, the three that fall at stage 2:

1. **`iv.py` and `uxm_relock.py` on the core API plus wiring config, no register paths** — as a
   written mapping (Appendix B). `iv.py`'s nine names and `uxm_relock.py`'s forty-four all map;
   the four private reaches (`_caget`, `_caput`, `_cryo_root`, `_bad_mask`) reduce to one catalog
   name (`band[b].ops.in_progress`) and one config field. Residue: none.
2. **All 156 names classified on three axes, counts published, ambiguous names listed** —
   Appendix A; `scripts/stage2_classification_check.py` asserts every artifact name appears exactly
   once, every cell is in its vocabulary, and the counts table matches.
3. **The grep gate**: no `is_rfsoc`, `tes`, `bias_group`, `pA_per_phi0` under `cryodaq`; no `614.4`,
   bare `128`/`512`, or `2.4` either. Lands with the package.

Plus the two CI tests — compat contract (§8) and import-direction (§7.3) — and the skeleton's
hardware conditions from the proposal's stage-2 entry, which this document does not restate.

**Where they stand.** Conditions 1 and 2 were met by Appendices A and B and the
classification checker, on `58ff4e41`, and the 2026-09-11 revision does not touch them. Condition 3
lands with the package and is one of the eight checks in `tests/cryodaq/check_boundaries.py`; two of
them are the revision's: **a register path appears only under `cryodaq.platform`** — now the main
boundary rule, and what "the client does not know the register map" means mechanically — and
**`cryodaq.platform` imports neither rogue nor pyrogue**, which is what makes the map a map and not a
second client. The rule revision 1 carried, that pyrogue is imported only under `cryodaq.platform`, is
**inverted and gone**: the client is a rogue client, so confining rogue to one module was an artificial
boundary that bought a wrapper layer and no separation (Tristan, 2026-09-11). Two of the eight do not
read the package but run it, which is the only way to show that it needs nothing but Python until a
session is opened. A second script joined the first on 2026-09-14, on the platform layer's own
decisions: which platform a system is identified as, and which indices of a scope a tree has.

Of the two CI tests, **import-direction is in place** as one of those rules; the **compat contract test
is deferred to stage 3**, because the shim it would test does not exist yet — the contract is specified
in §8 and nothing generates it here. That deferral is recorded with its discharge conditions in the
stage report, not resolved by this document.

The hardware conditions the proposal's stage-2 entry sets are met on the live slot-4 carrier at firmware
`0x2050000`: the platform-resolved endpoint (`crate:4`), 402 of 402 offered names resolving with none
unresolved, the witness set read and reproduced against its own report, the scopes discovered rather
than declared, and one operation invoked as a command with a write and a verified restore on two bands.
Two things are shown in emulation only and are stated as such in the stage report — **starting a
`Process`** under a bounded wait, which on this server would be a tuning run with nothing to tune, and
**attaching to a configured server** (the available server has never been set up, which is itself why
§2.2's refusal was dropped).

## 10. Open questions for reviewers

Only those that change a shape in this document. The proposal's wider questions stand.

Four of these were **answered by the 2026-09-11 revision** and are struck rather than deleted, so a
reviewer can see what became of them: *derived names* — dropped, they are functions over names the map
already has, not map entries (§3.1); *the error model* — reduced to three, and the argument for the
other five is now in §6's second table; *the provider protocol* — deferred with the whole server-side
seam (§7.3), so the question to SPECTRA about scopes is still live but is not stage 2's; *attenuators
core or optional* — no longer a question about a record, since a tree without them has no `uc`/`dc`
indices and the names are simply not offered (§7.1). The question *behind* it stands and is asked
below.

- **Session or connection?** One session per system (§2.3) is provisional. Anyone with a concrete
  multi-system or non-SMuRF use should say what the connection object would need to hold.
- **Attenuators, and RF front ends generally.** A tree without `MicrowaveMuxCore[*]` has no
  attenuator names, so nothing declares anything (§7.1). The question that remains is about the
  operations above them: SPECTRA, which of your tiers have programmable attenuation and where, and is
  any core operation entitled to assume it?
- **Where the map lives.** It is one module of data per hardware generation under `cryodaq.platform`,
  and it is the only place a register path appears (§3.1). Firmware team: should that table be
  generated from, or checked against, something cryo-det publishes — rather than maintained here and
  checked against a firmware tree by hand, which is what it does today? Nothing checks it against a
  firmware package in CI: no released package can build the tree that check would need, so the gate is
  deferred rather than in place.
- **The flux-ramp group homed in the modality** (`flux_ramp_setup`, `set_fixed_flux_ramp_bias`,
  `estimate_lms_freq`, `optimize_lms_delay`): SO, is anything in `uxm_setup`/`uxm_relock` calling
  these in a way that assumes they are unconditional core operations?
- **The error model** (§6) is now three exceptions. Objections to raise-never-`None`, to retiring the
  cryocard sentinels, or to leaving transport and operation failure as rogue reports them rather than
  giving each a name of our own?
- **A process's verdict is its `Message`** (§4.1). Rogue tells a client that a process stopped, not
  whether it succeeded, and a run shorter than one poll interval is indistinguishable from one that
  never started. Firmware team: is a status code on `pyrogue.Process` something rogue would take, or
  should each algorithm write a machine-readable result to a register of its own?
- **The link monitor's thread** (§2.3). rogue starts it as a non-daemon thread, so a client nobody
  stopped holds the interpreter open, and the only hook that runs early enough to stop it is
  CPython-internal. Firmware team: would rogue take `daemon=True` there? A background poller outliving
  the program it polls for is nobody's intent, and one word upstream would delete the workaround here
  and fix every other rogue client at the same time.
- **`save_state`**: stage 4 replaces it with the published description. Does anyone need a
  client-callable "write the server's state to disk" after that?
- **The per-band busy flag** (`band[b].ops.in_progress`, Appendix B): `uxm_relock.py` writes it to 0
  to recover from an aborted run. Keep it a writable state name, or make the recovery an operation
  (`band[b].ops.reset`) so that a Process can never be cleared from under itself? SO's call.
- **Scopes** (§7.3, deferred): SPECTRA, a LIM/MKID readout's operations would attach per band or per
  system as the µMUX ones do — is that enough, or does a fast-regime operation need a scope of its own?
  A new scope is a row of data in the map, so the cost is low, but the shape of the question is not.
- **The 13 ambiguous names** in Appendix A, each with its question in the *New home / note* column.
- **Map size.** §3.1's rule — a name enters only when a core operation or the shim needs it — is the
  guard against re-growing the register map. Is a rule enough, or should the map carry a budget the CI
  job enforces? It is 71 patterns today, against 216 register-name constants in today's client.

## Appendix A — classification of the compat surface

All 156 `S.*` names sodetlib touches (`artifacts/stage2/sodetlib_S_attrs_tpm-rogue6_04cfa27.txt`,
generated 2026-09-10 from sodetlib `tpm/rogue6` `04cfa27`), one row each. *Today* is the definition
line under `python/pysmurf/client/` at `58ff4e41` (for config properties, the `__init__` assignment).
Vocabularies: Layer `platform | core | application | utility`; Scope `µMUX | general`; Kind
`procedure | analysis | capability | access | state | utility` (§1). Names whose note begins
**Ambiguous** are the design questions; they are listed after the counts.

**The classification is unchanged by revision 2** — the layer, scope and kind of every name is the same,
and `scripts/stage2_classification_check.py` still passes on it. What revision 2 changes is the
*mechanism* the *New home / note* column names in a few rows. Where the revision replaced a call with
one that exists today the row now says the current one, because a reader trying the old one would get
an `AttributeError`: `sess.platform.raw(path)` has become `sess.root.getNode(path)` and
`sess.invoke(name)` has become `sess.call(name)`, in Appendix B too. Where what the row names is a
design object the revision *removed* rather than replaced, the row is left as it stood — it is the
record of what revision 1 proposed, and there is nothing yet to put in its place — and how to read it
is given once here:

- **`sess.platform.<capability>`** — a typed capability record, and §7.1 removed all eleven of them.
  Read the capability as a family of names and the field as the register it is read from: the *Capability*
  column of §7.1's inventory says which family, and nothing declares one. `platform.geometry.<field>`
  is the case with names already: `band[b].center_mhz` and its four siblings (§3.1).
- A **derived** name — `channel_to_freq`, `which_on`, `get_subband_from_channel` — is a function over
  names the map has, not a map entry, so the question in those rows' notes is answered by §3.1 rather
  than still open.

Everything else in the appendices, and in the sections the top of the document lists as unrevised,
reads as revision 1 wrote it.

| Name | Today | Layer | Scope | Kind | New home / note |
|---|---|---|---|---|---|
| `C` | `base_class.py:230` | platform | general | capability | `sess.platform.cryocard` (optional capability); the shim keeps the real `CryoCard` |
| `LOG_ERROR` | `base_class.py:68` | utility | general | utility | `sess.log` level constants |
| `LOG_INFO` | `base_class.py:72` | utility | general | utility | `sess.log` level constants |
| `LOG_USER` | `base_class.py:64` | utility | general | utility | `sess.log` level constants |
| `R_sh` | `smurf_config_properties.py:1142` | application | µMUX | state | TES wiring constant; application config |
| `_amplitude_scale` | `smurf_config_properties.py:141` | core | general | state | default tone power per band; session config |
| `_bad_mask` | `smurf_config_properties.py:177` | core | general | state | frequency exclusion mask for resonator search; session config |
| `_bands` | `smurf_config_properties.py:142` | core | general | state | bands to operate; `sess.description` |
| `_bias_group_to_pair` | `smurf_config_properties.py:176` | application | µMUX | state | TES wiring; application config |
| `_bias_line_resistance` | `smurf_config_properties.py:159` | application | µMUX | state | TES wiring; application config |
| `_caget` | `smurf_command.py:141` | core | general | access | `sess.get(name)`; the shim forwards a *path* string to `sess.root.getNode(path)` with today's `count`/`index` behaviour (decision 6). **Ambiguous:** the raw route is the whole tree — is that boundary enough? |
| `_caput` | `smurf_command.py:49` | core | general | access | `sess.set(name, value)`; shim forwards to `sess.root.getNode(path)` with today's `wait_done`. **Ambiguous:** as `_caget` |
| `_cryo_root` | `base_class.py:327` | platform | general | access | register path prefix — no public home (register paths live only in `cryodaq.platform`); shim returns the resolved string. **Ambiguous** |
| `_feedback_to_feedback_frac` | `smurf_tune.py:2470` | core | µMUX | state | flux-ramp modality derived value; set by `tracking_setup` |
| `_high_low_current_ratio` | `smurf_config_properties.py:160` | application | µMUX | state | TES wiring; application config |
| `_n_bias_groups` | `smurf_config_properties.py:175` | application | µMUX | state | TES wiring; application config |
| `_pA_per_phi0` | `smurf_config_properties.py:94` | application | µMUX | state | SQUID calibration; application config |
| `_pic_to_bias_group` | `smurf_config_properties.py:162` | application | µMUX | state | TES wiring; application config |
| `_rtm_slow_dac_bit_to_volt` | `base_class.py:249` | platform | general | capability | `sess.platform.bias_lines` DAC scale |
| `_rtm_slow_dac_data_reg` | `smurf_command.py:7320` | platform | general | access | register-name constant — no public home; shim keeps the string. **Ambiguous:** it is what sodetlib #482 builds paths from |
| `_rtm_slow_dac_nbits` | `base_class.py:246` | platform | general | capability | `sess.platform.bias_lines` DAC width |
| `_sodetlib_cfg` | `det_config.py:694 (sodetlib)` | application | general | state | sodetlib's own monkey-patch; **outside the contract** (decision 5); keeps working on any object that allows `setattr` |
| `all_off` | `smurf_util.py:3805` | core | general | procedure | `band_off` over every band |
| `amplitude_scale` | `smurf_config_properties.py:2022` | core | general | state | public face of `_amplitude_scale` |
| `analyze_psd` | `smurf_noise.py:1434` | core | general | analysis | client-side, over data |
| `bands` | `smurf_config_properties.py:1732` | core | general | state | public face of `_bands` |
| `base_dir` | `smurf_control.py:189` | utility | general | state | `sess.paths` |
| `bias_group_to_pair` | `smurf_config_properties.py:1040` | application | µMUX | state | public face of `_bias_group_to_pair` |
| `bias_line_resistance` | `smurf_config_properties.py:1098` | application | µMUX | state | public face of `_bias_line_resistance` |
| `channel_off` | `smurf_util.py:2194` | core | general | procedure | `channel_off(sess, band, channel)` |
| `channel_to_freq` | `smurf_util.py:2521` | core | general | access | derived name `band[b].channel[c].frequency_mhz`, resolved with platform geometry. **Ambiguous:** derived names are a catalog feature the spec has to allow |
| `check_adc_saturation` | `smurf_util.py:1717` | core | general | procedure | part of `check_ready` / raw-IQ capture |
| `check_jesd` | `smurf_util.py:2322` | platform | general | procedure | `check_ready(sess)` — JESD, timing |
| `date` | `smurf_control.py:211` | utility | general | state | `sess.paths` / session start time |
| `estimate_phase_delay` | `smurf_util.py:328` | core | general | procedure | `estimate_phase_delay(sess, band)` — named in the operations decision |
| `find_freq` | `smurf_tune.py:3473` | core | general | procedure | `find_freq(sess, band) -> sweep` |
| `flux_ramp_off` | `smurf_command.py:8034` | core | µMUX | procedure | flux-ramp modality operation |
| `flux_ramp_setup` | `smurf_tune.py:3173` | core | µMUX | procedure | flux-ramp modality operation (`rate_khz`, `fraction_full_scale`) |
| `freq_resp` | `base_class.py:240` | core | general | state | tune record; `last_tune(sess)` |
| `full_band_ampl_sweep` | `smurf_tune.py:3665` | core | general | procedure | the sweep behind `find_freq` |
| `full_band_resp` | `smurf_tune.py:583` | core | general | procedure | `full_band_response(sess, band) -> sweep` |
| `get_amplifier_biases` | `smurf_command.py:7905` | platform | general | capability | `sess.platform.amplifiers` |
| `get_amplitude_scale_array` | `smurf_command.py:1901` | core | general | access | `band[b].tone.amplitude` |
| `get_att_dc` | `smurf_command.py:4994` | platform | general | capability | attenuators. **Ambiguous:** the draft lists `set_attenuation` under core tones; RFSoC eval boards have none — proposed as an optional platform capability |
| `get_att_uc` | `smurf_command.py:4921` | platform | general | capability | as `get_att_dc` |
| `get_band_center_mhz` | `smurf_command.py:4303` | platform | general | capability | geometry: `platform.geometry.band_centers_mhz` |
| `get_center_frequency_array` | `smurf_command.py:3236` | core | general | access | `band[b].tone.frequency` |
| `get_center_frequency_mhz_channel` | `smurf_command.py:4670` | core | general | access | per-channel form of the above |
| `get_channel_frequency_mhz` | `smurf_command.py:4338` | platform | general | capability | geometry: channelizer bin width |
| `get_closest_subband` | `smurf_tune.py:1536` | core | general | analysis | arithmetic over `platform.geometry` |
| `get_cryo_card_ac_dc_mode` | `smurf_command.py:8792` | platform | µMUX | capability | cryocard relay (flux-ramp coupling) |
| `get_cryo_card_relays` | `smurf_command.py:8657` | platform | µMUX | capability | cryocard relays (TES bias) |
| `get_data_file_name` | `smurf_command.py:10420` | core | general | access | `stream.path` |
| `get_downsample_factor` | `smurf_command.py:10257` | core | general | access | stream rate (modality-selected) |
| `get_eta_mag_array` | `smurf_command.py:3447` | core | general | access | `band[b].eta.mag` |
| `get_eta_phase_array` | `smurf_command.py:3354` | core | general | access | `band[b].eta.phase` |
| `get_feedback_enable` | `smurf_command.py:3113` | core | general | access | `band[b].feedback.enable` |
| `get_feedback_enable_array` | `smurf_command.py:1980` | core | general | access | per-channel form |
| `get_feedback_end` | `smurf_command.py:3849` | core | µMUX | access | feedback window in the flux-ramp frame |
| `get_feedback_start` | `smurf_command.py:3790` | core | µMUX | access | as above |
| `get_filter_disable` | `smurf_command.py:10339` | core | general | access | stream filter stage |
| `get_flux_ramp_freq` | `smurf_command.py:8173` | core | µMUX | access | flux-ramp modality |
| `get_fpga_git_hash_short` | `smurf_command.py:5636` | platform | general | capability | `sess.platform.firmware` |
| `get_fraction_full_scale` | `smurf_tune.py:3303` | core | µMUX | access | flux-ramp modality |
| `get_lms_enable1` | `smurf_command.py:3907` | core | µMUX | access | flux-ramp demodulation |
| `get_lms_enable2` | `smurf_command.py:3959` | core | µMUX | access | flux-ramp demodulation |
| `get_lms_enable3` | `smurf_command.py:4011` | core | µMUX | access | flux-ramp demodulation |
| `get_lms_freq_hz` | `smurf_command.py:4133` | core | µMUX | access | flux-ramp demodulation frequency |
| `get_lms_gain` | `smurf_command.py:3684` | core | µMUX | access | tracking loop gain (flux-ramp demod) |
| `get_loop_filter_output_array` | `smurf_command.py:3139` | core | general | access | tracked frequency error |
| `get_number_channels` | `smurf_command.py:537` | platform | general | capability | geometry: channels per band |
| `get_rtm_slow_dac_volt_array` | `smurf_command.py:7496` | platform | general | capability | `sess.platform.bias_lines` |
| `get_streamdatawriter_datafile` | `smurf_command.py:6005` | core | general | access | `stream.path` |
| `get_subband_from_channel` | `smurf_util.py:2619` | core | general | analysis | arithmetic over `platform.geometry` |
| `get_tes_bias_bipolar` | `smurf_util.py:2920` | application | µMUX | access | bias group → platform bias lines |
| `get_tes_bias_bipolar_array` | `smurf_util.py:2971` | application | µMUX | access | as above |
| `get_timestamp` | `smurf_control.py:740` | utility | general | utility | client wall clock; not an interface concern. **Ambiguous:** declared gap — proposed: no public home, shim only |
| `get_tone_frequency_offset_mhz` | `smurf_command.py:3173` | core | general | access | `band[b].tone.frequency_offset` |
| `high_low_current_ratio` | `smurf_config_properties.py:1181` | application | µMUX | state | public face of `_high_low_current_ratio` |
| `lms_freq_hz` | `smurf_config_properties.py:1442` | core | µMUX | state | flux-ramp modality config |
| `load_tune` | `smurf_tune.py:4289` | core | general | procedure | `load_tune(sess, path)` — writes the record to hardware |
| `log` | `base_class.py:87` | utility | general | utility | `sess.log` |
| `make_sync_flag` | `smurf_tune.py:4721` | core | µMUX | analysis | flux-ramp sync flags from stream data |
| `name` | `smurf_control.py:217` | utility | general | state | `sess.paths` / session identity |
| `output_dir` | `smurf_control.py:112` | utility | general | state | `sess.paths.output` |
| `overbias_tes_all` | `smurf_util.py:3080` | application | µMUX | procedure | registered application operation `overbias_tes` |
| `pA_per_phi0` | `smurf_config_properties.py:401` | application | µMUX | state | public face of `_pA_per_phi0` |
| `play_sine_tes` | `smurf_util.py:3912` | application | µMUX | procedure | application operation over the platform's `bias_lines.waveform` capability. **Ambiguous:** RTM arb waveform, see `set_rtm_arb_waveform_*` |
| `plot_dir` | `smurf_control.py:192` | utility | general | state | `sess.paths.plot` |
| `pub` | `base_class.py:126` | utility | general | utility | `sess.pub`; stays a pysmurf `Publisher` on the shim (decision 5) |
| `read_adc_data` | `smurf_util.py:1773` | core | general | procedure | raw-IQ capture (the raw-IQ modality's primitive) |
| `read_stream_data` | `smurf_util.py:1183` | core | general | analysis | `read_stream(path) -> timestreams` |
| `relock` | `smurf_tune.py:1912` | core | general | procedure | `relock(sess, band)` |
| `rtm_spi_max_root` | `base_class.py:214` | platform | general | access | register path prefix — no public home; shim keeps the string |
| `run_serial_eta_scan` | `smurf_command.py:1135` | core | general | procedure | `eta_scan` — stage-1 Process |
| `run_serial_gradient_descent` | `smurf_command.py:1161` | core | general | procedure | `gradient_descent` — stage-1 Process |
| `save_state` | `smurf_command.py:1236` | core | general | procedure | rogue `SaveState` (#1015). **Ambiguous:** replaced by the published description at stage 4 — does a client-callable form survive? |
| `save_tune` | `smurf_tune.py:4273` | core | general | procedure | `save_tune(sess) -> path` — the record is server-authoritative |
| `set_50k_amp_gate_voltage` | `smurf_command.py:7972` | platform | general | capability | `sess.platform.amplifiers` |
| `set_amp_drain_voltage` | `smurf_command.py:7715` | platform | general | capability | `sess.platform.amplifiers` |
| `set_amp_gate_voltage` | `smurf_command.py:7569` | platform | general | capability | `sess.platform.amplifiers` |
| `set_amplifier_bias` | `smurf_command.py:7992` | platform | general | capability | `sess.platform.amplifiers` |
| `set_amplitude_scale_array` | `smurf_command.py:1873` | core | general | access | `set_tone_amplitudes` is sugar over `sess.set('band[b].tone.amplitude', …)` |
| `set_att_dc` | `smurf_command.py:4956` | platform | general | capability | as `get_att_dc` |
| `set_att_uc` | `smurf_command.py:4883` | platform | general | capability | as `get_att_dc` |
| `set_band_delay_us` | `smurf_command.py:2815` | core | general | access | `band[b].delay_us` (a stage-1 witness) |
| `set_center_frequency_array` | `smurf_command.py:3205` | core | general | access | `set_tone_frequencies` is sugar over `sess.set` |
| `set_center_frequency_mhz_channel` | `smurf_command.py:4641` | core | general | access | per-channel form |
| `set_dac_axil_addr` | `smurf_command.py:6427` | platform | general | capability | RTM arb-waveform addressing. **Ambiguous:** see `set_rtm_arb_waveform_*` |
| `set_downsample_factor` | `smurf_command.py:10219` | core | general | access | `set_downsample(sess, factor)` |
| `set_downsample_filter` | `smurf_util.py:3451` | core | general | access | `set_filter(sess, …)`; coefficient design is client analysis, the write is access |
| `set_downsample_mode` | `smurf_command.py:10171` | core | general | access | stream rate mode |
| `set_eta_mag_array` | `smurf_command.py:3412` | core | general | access | `band[b].eta.mag` |
| `set_eta_phase_array` | `smurf_command.py:3322` | core | general | access | `band[b].eta.phase` |
| `set_feedback_enable` | `smurf_command.py:3086` | core | general | access | `enable_feedback` is sugar over `sess.set` |
| `set_feedback_enable_array` | `smurf_command.py:1950` | core | general | access | per-channel form |
| `set_filter_disable` | `smurf_command.py:10324` | core | general | access | stream filter stage |
| `set_fixed_flux_ramp_bias` | `smurf_tune.py:3108` | core | µMUX | procedure | flux-ramp modality operation. **Ambiguous:** declared gap — proposed home is the modality, not the core |
| `set_fixed_tone` | `smurf_util.py:4038` | core | general | procedure | finds the channel for a frequency, then sets it |
| `set_flux_ramp_dac` | `smurf_command.py:6898` | core | µMUX | access | flux-ramp modality |
| `set_gradient_descent_converge_hz` | `smurf_command.py:910` | core | general | access | parameter of the `gradient_descent` Process |
| `set_gradient_descent_max_iters` | `smurf_command.py:744` | core | general | access | as above |
| `set_gradient_descent_step_hz` | `smurf_command.py:964` | core | general | access | as above |
| `set_hemt_gate_voltage` | `smurf_command.py:7977` | platform | general | capability | `sess.platform.amplifiers` |
| `set_lms_enable1` | `smurf_command.py:3879` | core | µMUX | access | flux-ramp demodulation |
| `set_lms_enable2` | `smurf_command.py:3933` | core | µMUX | access | flux-ramp demodulation |
| `set_lms_enable3` | `smurf_command.py:3985` | core | µMUX | access | flux-ramp demodulation |
| `set_lms_gain` | `smurf_command.py:3654` | core | µMUX | access | flux-ramp demodulation |
| `set_logfile` | `base_class.py:305` | utility | general | utility | `sess.log` |
| `set_mode_ac` | `smurf_util.py:3267` | platform | µMUX | capability | cryocard AC/DC relay |
| `set_mode_dc` | `smurf_util.py:3242` | platform | µMUX | capability | cryocard AC/DC relay |
| `set_read_all` | `smurf_command.py:677` | platform | general | access | rogue root `ReadAll`; `sess.description` refresh |
| `set_rtm_arb_waveform_continuous` | `smurf_command.py:6361` | platform | general | capability | RTM arb waveform. **Ambiguous:** declared gap — proposed as optional capability `bias_lines.waveform`, with `play_sine_tes` the application operation over it |
| `set_rtm_arb_waveform_enable` | `smurf_command.py:6547` | platform | general | capability | as above |
| `set_rtm_arb_waveform_lut_table` | `smurf_command.py:6241` | platform | general | capability | as above |
| `set_rtm_arb_waveform_timer_size` | `smurf_command.py:6455` | platform | general | capability | as above |
| `set_rtm_slow_dac_enable` | `smurf_command.py:7214` | platform | general | capability | `sess.platform.bias_lines` |
| `set_rtm_slow_dac_volt_array` | `smurf_command.py:7475` | platform | general | capability | `sess.platform.bias_lines` |
| `set_stream_enable` | `smurf_command.py:2181` | core | general | access | `stream(sess, …)` context |
| `set_synthesis_scale` | `smurf_command.py:4405` | core | general | access | `band[b].tone.synthesis_scale` |
| `set_tes_bias_bipolar` | `smurf_util.py:2785` | application | µMUX | access | bias group → platform bias lines |
| `set_tes_bias_bipolar_array` | `smurf_util.py:2840` | application | µMUX | access | as above |
| `set_tes_bias_high_current` | `smurf_util.py:3158` | application | µMUX | procedure | relay flip with DAC rescale — the operation sodetlib #482 bypasses; gets a public home as a registered application operation |
| `set_tes_bias_low_current` | `smurf_util.py:3209` | application | µMUX | procedure | as above |
| `setup` | `smurf_control.py:322` | core | general | procedure | `setup(sess, cfg)` — the one operation that takes a config |
| `setup_notches` | `smurf_tune.py:3892` | core | general | procedure | `setup_notches(sess, band, assignment)` |
| `stream_data_off` | `smurf_util.py:1160` | core | general | procedure | `stream(sess, …)` context exit |
| `stream_data_on` | `smurf_util.py:984` | core | general | procedure | `stream(sess, …)` context entry |
| `take_debug_data` | `smurf_util.py:43` | core | general | procedure | raw-IQ capture |
| `take_stream_data` | `smurf_util.py:865` | core | general | procedure | `with stream(sess, …)` plus a timed wait |
| `toggle_feedback` | `smurf_util.py:2128` | core | general | procedure | feedback off, wait, on |
| `tracking_setup` | `smurf_tune.py:2494` | core | µMUX | procedure | today's is flux-ramp specific; becomes `tracking_setup(sess, band, modality, …)` |
| `tune_file` | `smurf_tune.py:4283` | core | general | state | `last_tune(sess)` — the tune record (§4.2), not a session attribute (§2.4) |
| `unset_fixed_flux_ramp_bias` | `smurf_tune.py:3081` | core | µMUX | procedure | flux-ramp modality operation |
| `which_bays` | `smurf_util.py:2437` | platform | general | capability | `sess.platform.bays` |
| `which_on` | `smurf_util.py:2110` | core | general | access | derived name `band[b].channels_on`. **Ambiguous:** declared gap — derived names, as `channel_to_freq` |

### The one-level descent — `S.C.*` and `S.pub.*`

The attributes of pysmurf's sub-objects that sodetlib reaches
(`artifacts/stage2/sodetlib_S_sub_attrs_tpm-rogue6_04cfa27.txt`), inside the contract by decision 5.

| Sub-attribute | Today | Layer | Scope | Kind | Note |
|---|---|---|---|---|---|
| `C.do_write` | `cryo_card.py:107` | platform | general | capability | raw PIC register write; reached by sodetlib #482. Public home: `platform.cryocard` low-level access |
| `C.get_fw_version` | `cryo_card.py:347` | platform | general | capability | card revision and liveness (`(0,0,0)` sentinel today) |
| `C.list_of_c02_amps` | `cryo_card.py:68` | platform | general | capability | amplifier inventory per card revision → `platform.amplifiers` |
| `C.list_of_c02_and_c04_amps` | `cryo_card.py:70` | platform | general | capability | as above |
| `C.list_of_c04_amps` | `cryo_card.py:69` | platform | general | capability | as above |
| `C.read_ps_en` | `cryo_card.py:327` | platform | general | capability | amplifier drain enables → `platform.amplifiers` |
| `C.relay_address` | `cryo_card.py:55` | platform | µMUX | capability | PIC register address constant; used with `do_write` |
| `C.write_optical` | `cryo_card.py:382` | platform | general | capability | optical TX enables |
| `C.write_ps_en` | `cryo_card.py:308` | platform | general | capability | amplifier drain enables → `platform.amplifiers` |
| `pub._action` | `pub.py:121` | utility | general | utility | read by sodetlib `util.py:204`; private but pysmurf's, so preserved (decision 5) |
| `pub._action_ts` | `pub.py:122` | utility | general | utility | as above |
| `pub.publish` | `pub.py:158` | utility | general | utility | `sess.pub.publish` |
| `pub.register_file` | `pub.py:186` | utility | general | utility | `sess.pub.register_file` — 55 of sodetlib's 69 `S.pub` uses |

### Counts

Each axis partitions the 156 top-level names; the sub-attributes are not counted here.

| Axis | Value | Count |
|---|---|---|
| Layer | platform | 35 |
| Layer | core | 89 |
| Layer | application | 20 |
| Layer | utility | 12 |
| Scope | µMUX | 45 |
| Scope | general | 111 |
| Kind | procedure | 32 |
| Kind | analysis | 5 |
| Kind | capability | 30 |
| Kind | access | 56 |
| Kind | state | 26 |
| Kind | utility | 7 |

### The ambiguous names

Thirteen, by name — the classification pass's real output:

- `_caget`, `_caput`, `_cryo_root`, `_rtm_slow_dac_data_reg` — register-path access from the client.
  The revision answers the *where* and leaves the question: there is no accessor of ours, expert or
  otherwise, because `sess.root` is the tree and `getNode(path)` is rogue's own (§3.1), and the shim
  forwards to that; no path-string constants in the public interface. The question is whether handing
  back the whole tree is boundary enough, and what in `smurf_cmd.py` / `scratch/` relies on the
  constants.
- `channel_to_freq`, `which_on` — derived names; need catalog resolvers (§3.1, §10).
- `get_att_dc` (and `_uc`, `set_att_*`) — attenuators as optional capability (§7.1, §10).
- `set_rtm_arb_waveform_continuous` (and the other three), `set_dac_axil_addr`, `play_sine_tes` —
  the RTM arbitrary waveform: proposed optional capability `bias_lines.waveform` with
  `play_waveform` as the application operation over it.
- `set_fixed_flux_ramp_bias` — homed in the flux-ramp modality, not the core.
- `save_state` — replaced at stage 4; survival of a client-callable form is open.
- `get_timestamp` — client wall clock; proposed dropped from the interface, kept on the shim.

## Appendix B — `iv.py` and `uxm_relock.py` on the core API

*Status: both mapped (step 4, 2026-09-11). Residue: none — no register path and no private
reach remains; one flag is renamed on the way (`in_progress`, below).*

`sodetlib/operations/iv.py` uses nine `S.*` names. Every one is application-layer or utility, which
is the expected result for a TES procedure:

| `iv.py` name | Lands as |
|---|---|
| `overbias_tes_all` | `overbias_tes(sess, bias_groups, …)` — registered application operation |
| `set_tes_bias_bipolar`, `set_tes_bias_bipolar_array`, `get_tes_bias_bipolar_array` | application access over `platform.bias_lines`, via the bias-group wiring config |
| `high_low_current_ratio`, `_n_bias_groups` | application configuration |
| `log`, `LOG_INFO`, `pub` | `sess.log`, `sess.pub` |

`sodetlib/operations/uxm_relock.py` uses forty-four `S.*` names across its five functions
(`reload_tune`, `run_grad_descent_and_eta_scan`, `get_full_band_sweep`, `plot_channel_resonance`,
`uxm_relock`). All forty-four have Appendix A rows; grouped by where they land:

| `uxm_relock.py` name | Lands as |
|---|---|
| `relock`, `setup_notches`, `load_tune`, `save_tune`, `full_band_ampl_sweep`, `all_off` | core procedures of §4.2 (`relock(sess, band)`, `setup_notches(sess, band, assignment)`, `load_tune(sess, path)`, `save_tune(sess) -> path`, the sweep behind `find_freq`, `band_off` over all bands) |
| `run_serial_gradient_descent`, `run_serial_eta_scan` | `sess.call("band[b].ops.gradient_descent")`, `…eta_scan` — the stage-1 Processes, registered by the first provider (§7.3) |
| `set_gradient_descent_{max_iters, converge_hz, step_hz}` | `Param`s of `gradient_descent`; set-then-run keeps working through the parameter nodes (§7.3, *Inputs*) |
| `{get,set}_{amplitude_scale, center_frequency, eta_mag, eta_phase, feedback_enable}_array`, `get_tone_frequency_offset_mhz`, `set_synthesis_scale`, `set_band_delay_us`, `{get,set}_feedback_enable` | `sess.get`/`sess.set` on `band[b].tone.amplitude`, `.tone.frequency`, `.tone.frequency_offset`, `.eta.mag`, `.eta.phase`, `.feedback.enable`, `.tone.synthesis_scale`, `.delay_us` |
| `set_downsample_factor`, `set_filter_disable` | `sess.set` on the stream names (`set_downsample(sess, factor)`, filter stage) |
| `get_band_center_mhz`, `get_subband_from_channel`, `channel_to_freq` | `platform.geometry.band_centers_mhz`; arithmetic over `platform.geometry`; the derived name `band[b].channel[c].frequency_mhz` (§3.1, ambiguous in Appendix A) |
| `set_att_uc`, `set_att_dc` | optional capability `attenuators` (§7.1); `uxm_relock` runs only where it is present |
| `set_mode_dc`, `set_mode_ac`, `set_rtm_arb_waveform_enable` | optional capabilities `cryocard` and `flux_ramp`; the modality owns the flux-ramp group (§7.2) |
| `amplitude_scale`, `tune_file`, `freq_resp` | the tune config field, and `last_tune(sess)` for the record — none of the three is a session attribute (§2.4) |
| `_bad_mask` | the resonator-search exclusion mask — a field of the tune configuration (§5); the shim's property today |
| `_cryo_root(b) + 'etaScanInProgress'` via `_caget` (line 98) and `_caput(…, 0)` (100, 125) | `sess.get("band[b].ops.in_progress")`; `sess.set("band[b].ops.in_progress", 0)` to clear a stale flag. `_cryo_root` disappears: no path is built on the client |
| `log` | `sess.log` |

The four private reaches therefore reduce to one catalog name and one config field. The name is
`band[b].ops.in_progress`, **not** `band[b].ops.eta_scan.in_progress` as first written: the flag
is a per-band mutex set and cleared by all four stage-1 Processes and by `runEtaScan`, so a
per-operation name would misdescribe it (§7.3, *Progress and status*). Its two writes in
`uxm_relock.py` clear a flag left set by an aborted run; whether that deserves an operation of its
own (`band[b].ops.reset`) rather than a writable state name is in §10. Nothing in either file
needs a register path, `_cryo_root`, or a name outside Appendix A: the residue is empty, which is
proposal boundary check 1 as a written mapping. Rewriting the two files is not stage 2's work; they
keep running unchanged through the shim (§8).

## Appendix C — deprecation notes for the shim

Not test failures (decision 6). Filled as the classification settles; the first entries are the six
names above with no public home — `_caget`, `_caput`, `_cryo_root`, `_rtm_slow_dac_data_reg`,
`rtm_spi_max_root`, `get_timestamp` — and the `None`-when-offline return of `_caget`.
