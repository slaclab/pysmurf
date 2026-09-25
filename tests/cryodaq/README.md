# Cryodaq test scripts

## Description

These scripts check the `cryodaq` client: that it keeps to its layer boundaries, that its platform
maps identify a system and enumerate what it has, and that every semantic name it offers reaches the
register, command or process its map says it does. None of them needs hardware.

They follow the convention of the [core test scripts](../core/README.md): each is a standalone
executable that prints one `ok` or `FAIL` line per check and exits non-zero if any check failed.

## Test scripts

### check_boundaries.py

This script checks the layer boundaries of [cryodaq](../../python/cryodaq). Ten checks: six read the
package as source, one reads what it carries that is not source, two run it, and the last is the
script's own selftest. None needs rogue or a CryoDet package, so this runs anywhere Python does.

The rules five of the source checks enforce — the first is that there is a package to read at all, and
the second covers the two import rules below it:

* **Import direction** — nothing under `cryodaq` imports `pysmurf`, `smurf` or `sodetlib`: the
  readout core never imports an application. The client reaches the platform layer through the
  `cryodaq.platform` package rather than one of its map modules.
* **The maps import no rogue** — a map is data, and the lookup over it is arithmetic on strings.
  A map module that reached for a tree would be doing the client's work in the wrong layer.
* **No application names** — `is_rfsoc`, `tes`, `bias_group` and `pA_per_phi0` appear nowhere under
  `cryodaq`. Which detectors are wired where belongs to the application above it, and branching on
  platform identity is what a per-platform map exists to avoid.
* **No geometry literals** — the channel counts, digitizer rates and bandwidths of one particular
  firmware are read from the tree, not written into the client.
* **Register paths only in `cryodaq.platform`** — a rogue register path appears only in the platform
  maps. This is the firmware/software boundary: it is what "the client does not know the register
  map" means mechanically.

A seventh check applies that last rule to everything the package carries that is *not* source. The
rule is enforced by reading string constants out of modules, so a register map shipped as data would
satisfy it without being subject to it — and a register map is exactly the kind of thing that is
easier to ship as data. Every non-source file outside the platform package is therefore read as text
and held to the same rule; a file that is not UTF-8 text cannot be read for paths, so it is reported
rather than passed over — the package ships no such file today, and one appearing would be the one
way to carry a path this check cannot see.

The sixth source check watches a different seam: the null publisher a session falls back to takes the
same parameter *names* as pysmurf's real `Publisher`, compared by parsing both. A caller that passes
them by keyword — sodetlib does — would get a `TypeError` from a stand-in that renamed them, so the
names are interface and not detail. It reads source because this script runs where pysmurf's own
plotting dependencies are not installed.

The two that run the package watch the same boundary from the other side: importing it, resolving an
endpoint, building the capture paths and looking up a map all work with nothing installed, and
`connect` judges every argument it can judge without a tree — a target that does not parse, a
platform no map goes by — before it reaches for rogue. The request deadline is not one of them: its
range belongs to the transport, so rogue refuses an unusable one and this client does not check it
twice. Both hold wherever this runs; where rogue is absent, as in CI, they also show that the package
does not quietly need it.

The selftest runs the source rules over a synthetic package that breaks each of them and fails if any
rule passes it. A boundary check that cannot fail is not evidence.

### check_platform_map.py

This script checks what the platform layer decides: which platform a system is, and which indices of
an indexed scope a tree has. Both are answered from values the caller supplies, so both are checked
without building a tree — it needs no rogue and runs in a second.

* **Identification** — a platform is the firmware it runs. Each map lists the firmware image names it
  covers, the name is taken from the build stamp, and firmware matching no map is refused quoting what
  it read and listing what is known. A system whose firmware cannot say what it is — an emulated
  register space reads as zeros — is refused too, and its platform is named instead. That is a lookup
  by name rather than a second way through identification: the two are separate functions, and the
  checks hold them apart by asserting that identification only ever reads the tree.
* **Name syntax** — a name that carries whitespace, at either end, is not a name: every character
  belongs to it, and one that did not would be formatted into a register path.
* **Scope enumeration** — gapped, sparse, empty and full index ranges. A firmware mask may leave an
  index out and keep a higher one, so a gap does not end a scope: collapsing one silently drops real
  hardware out of every name listing and witness that follows.
* **Caching a parse does not share its indices** — a name parses to the same answer every time, so the
  answer is kept; what a cache risks is handing two callers the same dictionary, where one of them
  changing it would corrupt what the other reads. A malformed name is also required to keep failing
  rather than being answered from a cached exception.
* **What the maps share** — the platforms of one generation share every register they can, and differ
  only by the hardware one of them does not carry: a name in both maps means the same register in
  both, and none is offered by the platform with fewer devices and not by the other. No two claim the
  same firmware.
* **A platform offers no name it cannot resolve** — every pattern fills to a path with no placeholder
  left, and every scope a pattern carries is one that platform declares. A name whose scope belongs to
  another platform's map would otherwise raise on its first call rather than at import.

### check_catalog_resolves.py

This script checks the register names against the firmware itself: every name is resolved through the
platform layer — the same lookup a running client does — and every resulting path is looked up in a
dump of the nodes a firmware package defines. The map is the contract between firmware and software,
so an entry that has drifted fails a build here instead of a night on a crate. It reads its dumps out
of the repository, so it needs nothing installed and runs where rogue is absent.

The names the client reaches are read from its source with `ast` rather than from a table beside it.
The accessors are hand-written, so the name each one resolves is in the call; a table of them would be
a second description of the same code, and the failure it invites is the table going stale while both
it and the code still pass. The reader refuses a file it cannot recognise at all, so a parser that
stopped matching fails here rather than quietly checking nothing.

* **Each platform offers only names its firmware has** — every name in a platform's map has to reach
  a register on that platform. This is the same assertion for both rather than a special case for
  one: a map is a statement of what a platform has, so a platform whose converters share the FPGA's
  die does not offer the front-end names at all, and asking for one is an unresolved name rather than
  a path that reaches nothing. Excluded are the tuning processes and the server's own configuration
  procedure, whose nodes the *server* attaches: a dump records what a firmware package defines, so
  those are absent from one by construction and their absence would say nothing.
* **The platforms differ by the hardware one lacks** — which names are carrier-only is *derived* from
  the two maps, not listed in this file, and then required to hold against the firmware: each must
  resolve on the carrier and not on the other. Both directions are asserted, because two maps
  offering the same names would make every comparison here vacuous, and a name offered by the
  platform with fewer devices and not by the carrier would mean the sharing is backwards.
* **The two dumps are different trees** — asserted directly, and asserted to differ by the per-bay
  front end rather than merely to differ, because a comparison of a tree against itself would pass
  every check above and prove nothing.
* **The client reaches only names the map resolves** — the names are read out of the client's own
  accessor calls, so what is checked is the code rather than a description of it, and a name that does
  not resolve is a method that raises the first time it is called. Every module under the client
  package is read — a wait in the tuning code or a poll in the utilities goes through the same helpers
  as an accessor — and `_wait_for` counts as a read. Only this direction fails: the map deliberately
  carries names no accessor reaches, because the operations use those directly.
* **The client writes no register the firmware makes read-only** — whether a register may be written
  is the firmware's statement, in the access mode it declares, so it is read from the dump rather than
  from anything describing the dump. The tree refuses such a write at run time on a good day; this is
  for the day it does not.
* **The map declares the kind the firmware declares** — a name the map calls a value is a value in the
  firmware, and a command a command. Resolution cannot catch this: the path is right and the node is
  there, so a value that is really a command surfaces only when something tries to write what has to
  be called. Rogue writes a command out as a write-only `int`, which is what makes it checkable from a
  dump; only that shape is judged, and the nodes it cannot tell apart are left alone rather than
  guessed at.

The dumps under `fixtures/` are pruned: a full one is some nine megabytes, almost all of it the
per-channel registers of eight bands. A row is kept when the device it sits in is one a name reaches,
with every index of that device retained, so both generations can still be told apart. They are
rebuilt by `prune_fixtures.py` from full dumps that `validate_client_emulated.py --dump-tree` writes
(its header gives the three commands); it records their provenance beside them and refuses to write a
pair that resolves differently from the full dumps — a pruned fixture that changed an answer would be
a smaller tree that happens to pass. Rerun it after any map change that reaches a new device.

The pruned dumps are the fallback that keeps the check meaningful on a fork's pull request, where no
token is available. Checking a *released* package is a second mode, `--package-dump`, run by the
`test-server` job: it downloads the release asset, builds a tree from it with
`dump_released_tree.py` (which needs rogue, hence that job rather than this one), and resolves the map
against it. Currently **v2.5.1**, the version deployed on the reference crate; all **180** names under
the firmware's own tree resolve against it.

The two modes ask different questions and the difference matters. A dump built from a package has no
**server-added subtrees** — `SmurfProcessor`, the two stream writers, `SmurfApplication`, the capture
receivers, `setDefaults` and `Ready` are all attached by `pysmurf.core.roots.Common` at start-up, so a
package has never heard of them. `--package-dump` therefore excludes those names and requires every
name under `AMCc.FpgaTopLevel` to resolve. Without that distinction the released-package run reports
about 54 false absences and a real one would be lost among them. `SERVER_ADDED_SUBTREES` names them,
and a selftest case asserts the exclusion covers exactly those subtrees and nothing under
`FpgaTopLevel` — a rule that grew to cover the firmware tree would excuse the absences it exists to
find.

What neither mode shows is that the names match every release, rather than the one checked. The map
claims a firmware *family* (`TAGS = ('MicrowaveMuxBpEthGen2',)`) and not a version range, so
"supported" is not yet a checkable statement; recording a range is the prerequisite for widening this.

### check_sodetlib_contract.py

This script checks that the client still offers everything sodetlib calls on it.

sodetlib drives this client from another repository and another organisation, so its use of the client
cannot be found by reading this one. A refactor here can remove a method nothing here calls and break an
observatory's analysis code, with nothing failing until someone runs it — so the surface sodetlib depends
on is frozen in `sodetlib_contract.json` and checked on every build.

What the freezing buys is that a removal becomes a **decision**. A name dropped from the contract appears
in a diff with a commit message saying why; a name dropped without touching the contract fails here. That
asymmetry is the point: it is easy to remove something by accident and hard to remove it deliberately
without noticing.

* **Every public name sodetlib calls is still offered** — 71 of them, measured from sodetlib rather than
  listed by hand. The check does not judge whether a removal is right, only that the contract was edited
  to say so.
* **Every private name it reaches is still there** — 14 underscore-prefixed attributes and methods, with a
  weaker promise: these are not interface and may go, but going needs the matching sodetlib change and
  this entry removed. Recording them is also how the list of what sodetlib has to stop using exists at
  all, rather than being rediscovered each time.
* **The contract names nothing the client never had** — an entry the client does not define means the
  measurement read something else, most likely a helper sodetlib defines itself, which would quietly
  weaken every assertion above. One such name is recorded as *not ours* and required to stay that way.
* **A deliberately dropped name stays dropped, with a reason** — so the dropped list cannot silently stop
  describing anything.

The surface is read from source with `ast`, as the union of the client's mixins, because `SmurfControl` is
assembled from eight of them and no single class holds it. Reading source rather than importing is what
lets this run on the bare runner beside the other checks here, where the client's plotting stack is absent.

Seven selftest cases drive each check with input it must refuse, including a client whose mixins cannot be
read at all — a contract check that passes by finding nothing is worse than none.

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
being handed the answer. Five further checks cover the path around it: a tree with a blank stamp is
refused, a declared platform connects anyway, a platform name that does not exist is refused, a
request deadline does not stop a working session, and a deadline the transport cannot express is
refused — by rogue, which is where that range lives. Each runs one at a time, before the session below
opens: pyrogue caches one client per address and port, so two sessions on one endpoint are one
transport — closing either closes both — and the client takes one deadline, whichever connected last.

The eighteen checks over that session cover the map against the tree (every name the map offers
resolves; each node is the kind the map declares; the twenty contract names are present on every band;
the witness registers read back; the indexed scopes are the ones this tree has) and the session itself
(what the server says it is; read and write by name, whole and by array index; a name the tree declares
read-only refused, and the value read back to show the refusal was the only thing that stopped it; a
value changed on the tree underneath the session seen by the next read, which is what every bounded
wait rests on; a command; a process under a bounded wait; the whole tree still reachable through
`session.root`; a wrong name and a wrong kind each refused with an exception that says which; a
session a program never closed still letting the interpreter exit, which is checked in a child process
because what it asserts is an exit). The
per-band ones work on a band
the session reports rather than on band 0: a tree whose bands start higher is legal, and the run's
header line records which band was used.

The session the checks use finds a configured server, and the flag that says so is set on the tree
directly rather than through the client — the server owns that value and a client is refused it, the
same reason the build stamp goes in underneath the register too.

Run it once per platform: the ATCA carrier by default, the RFSoC with `--rfsoc`. The RFSoC firmware's
own package is a subclass of this one that does nothing but default `isRFSOC` on, so the flag builds
that platform's tree without needing its repository on the path; what the flag changes is the JESD and
signal-generator configuration, which is why that platform has no RF front end and no serial links to
one. Both are bay-indexed even so — the acquisition mux is per-bay on either, so what differs is what
sits *inside* a bay. The scope check names what each platform is expected to have and fails if either
changes.

What it cannot show for the RFSoC is that the firmware image names in its map are the ones a real
RFSoC reports — the stamp here is one this script wrote. Only a live read settles that.

It does not show that an operation does anything useful. Emulated memory reads back zeros, so a
tuning process has nothing to find; what is checked is the route from a name to the node, and from a
failure to an exception that names what could not be resolved.
