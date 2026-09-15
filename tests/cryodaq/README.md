# Cryodaq test scripts

## Description

These scripts check the `cryodaq` client: that it keeps to its layer boundaries, that its platform
maps identify a system and enumerate what it has, and that every semantic name it offers reaches the
register, command or process its map says it does. None of them needs hardware.

They follow the convention of the [core test scripts](../core/README.md): each is a standalone
executable that prints one `ok` or `FAIL` line per check and exits non-zero if any check failed.

## Test scripts

### check_boundaries.py

This script checks the layer boundaries of [cryodaq](../../python/cryodaq). Five rules parse the
package with `ast`; two more import it and use the part of it that is meant to work with nothing
installed. Neither needs rogue or a CryoDet package, so this runs anywhere Python does.

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

The last two watch the same boundary from the other side, by running the package instead of reading
it: importing it, resolving an endpoint, building the capture paths and looking up a map all work
with nothing installed, and `connect` judges its arguments — a target that does not parse, a deadline
that is not a duration — before it reaches for rogue. Both hold wherever this runs; where rogue is
absent, as in CI, they also show that the package does not quietly need it.

### check_platform_map.py

This script checks what the platform layer decides: which platform a system is, and which indices of
an indexed scope a tree has. Both are answered from values the caller supplies, so both are checked
without building a tree — it needs no rogue and runs in a second.

* **Identification** — a platform is the firmware it runs. Each map lists the firmware image names it
  covers, the name is taken from the build stamp, and firmware matching no map is refused quoting what
  it read and listing what is known. A system whose firmware cannot say what it is — an emulated
  register space reads as zeros — is refused too, and its platform has to be declared instead; that is
  the one way past identification and it is meant to look deliberate.
* **Scope enumeration** — gapped, sparse, empty and full index ranges. A firmware mask may leave an
  index out and keep a higher one, so a gap does not end a scope: collapsing one silently drops real
  hardware out of every name listing and witness that follows.
* **What the maps share** — the platforms of one generation carry the same register table and no two
  claim the same firmware.

### validate_client_emulated.py

This script drives a `cryodaq` session against an emulated firmware tree.

It builds an [EmulationRoot](../../python/pysmurf/core/roots/EmulationRoot.py) over a CryoDet
package — a checkout with `--cryo-det`, or a pyrogue ZIP with `--zip` — serves it on a local port,
and then connects to it with `cryodaq.connect` exactly as a client connects to a deployed server.
There is one route in and it is the real one: no in-process shortcut, so what runs here is the code
that talks to a crate.

The package has to be one whose tuning operations the server attaches rather than the firmware. A
package that defines them itself is refused before a tree exists, and every released package so far
defines them, so a release off the shelf cannot be used here: `--cryo-det` takes a checkout with them
removed and `--zip` a ZIP built from one.

Because an emulated register space reads back zeros, the tree reports no firmware and so has no
platform. The script writes the build stamp of the platform it is building into the emulated memory
before connecting, so that identification runs here the same way it runs against a crate instead of
being handed the answer. Seven further checks cover the path around it: a tree with a blank stamp is
refused, a declared platform connects anyway, a platform name that does not exist is refused, a
request deadline does not stop a working session, and a deadline that is not a duration is refused.
The last two are about that shared client: pyrogue caches one per address and port, so two sessions on
one endpoint are one transport and one transport policy — a second session that asks for a different
deadline changes the first one's and is told so, and a monitor another session stopped stays stopped,
which connecting also says. The rest run one at a time before the session below opens, since closing
either of two sessions on an endpoint closes both.

The fifteen checks over that session cover the map against the tree (every name the map offers
resolves; each node is the kind the map declares; the twenty contract names are present on every band;
the witness registers read back; the indexed scopes are the ones this tree has) and the session itself
(what the server says it is; read and write by name, whole and by array index; a command; a process
under a bounded wait; the whole tree still reachable through `session.root`; a wrong name and a wrong
kind each refused with an exception that says which). The per-band ones work on a band the session
reports rather than on band 0: a tree whose bands start higher is legal, and the run's header line
records which band was used.

Run it once per platform: the ATCA carrier by default, the RFSoC with `--rfsoc`. The RFSoC firmware's
own package is a subclass of this one that does nothing but default `isRFSOC` on, so the flag builds
that platform's tree without needing its repository on the path; what the flag changes is the JESD and
signal-generator configuration, which is why the RFSoC has no bays. The scope check names what each
platform is expected to have and fails if either changes.

What it cannot show for the RFSoC is that the firmware image names in its map are the ones a real
RFSoC reports — the stamp here is one this script wrote. Only a live read settles that.

It does not show that an operation does anything useful. Emulated memory reads back zeros, so a
tuning process has nothing to find; what is checked is the route from a name to the node, and from a
failure to an exception that names what could not be resolved.
