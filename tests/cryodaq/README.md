# Cryodaq test scripts

## Description

These scripts check the `cryodaq` client: that it keeps to its layer boundaries, and that every
semantic name it offers reaches the register, command or process its platform map says it does.
Neither needs hardware.

They follow the convention of the [core test scripts](../core/README.md): each is a standalone
executable that prints one `ok` or `FAIL` line per check and exits non-zero if any check failed.

## Test scripts

### check_boundaries.py

This script checks the layer boundaries of [cryodaq](../../python/cryodaq) statically. It parses the
package with `ast` and imports nothing from it, so it needs neither rogue nor a CryoDet package and
runs anywhere Python does.

Five rules, one check each:

* **Register paths only in `cryodaq.platform`** — a rogue register path appears only in the platform
  maps. This is the firmware/software boundary: it is what "the client does not know the register
  map" means mechanically.
* **The maps import no rogue** — a map is data, and the lookup over it is arithmetic on strings.
  A map module that reached for a tree would be doing the client's work in the wrong layer.
* **Import direction** — nothing under `cryodaq` imports `pysmurf`, `smurf` or `sodetlib`: the
  readout core never imports an application. The client reaches the platform layer through the
  `cryodaq.platform` package rather than one of its map modules.
* **No application names** — `is_rfsoc`, `tes`, `bias_group` and `pA_per_phi0` appear nowhere under
  `cryodaq`. Which detectors are wired where belongs to the application above it, and branching on
  platform identity is what a per-platform map exists to avoid.
* **No geometry literals** — the channel counts, digitizer rates and bandwidths of one particular
  firmware are read from the tree, not written into the client.

A sixth check is the script's own selftest: it runs all five rules over a synthetic package that
breaks each of them and fails if any rule passes it. A boundary check that cannot fail is not
evidence.

### validate_client_emulated.py

This script drives a `cryodaq` session against an emulated firmware tree.

It builds an [EmulationRoot](../../python/pysmurf/core/roots/EmulationRoot.py) over a CryoDet
package — a checkout with `--cryo-det`, or a released pyrogue ZIP with `--zip` — serves it on a
local port, and then connects to it with `cryodaq.connect` exactly as a client connects to a
deployed server. There is one route in and it is the real one: no in-process shortcut, so what runs
here is the code that talks to a crate.

The fourteen checks cover the map against the tree (every offered name resolves; each node is the
kind the map declares; the twenty contract names are present on every band; the witness registers
read back; the indexed scopes are the ones this tree has) and the session over the connection (what
the server says it is; read and write by name, whole and by array index; a command; a process under
a bounded wait; the whole tree still reachable through `session.root`; a wrong name and a wrong kind
each refused with an exception that says which).

Run it once per supported tree: the ATCA carrier by default, the RFSoC generation with `--rfsoc`.
Both generations use one map and their trees differ by omission, so the scope check names what each
is expected to have and fails if either changes.

It does not show that an operation does anything useful. Emulated memory reads back zeros, so a
tuning process has nothing to find; what is checked is the route from a name to the node, and from a
failure to an exception that names what could not be resolved.
