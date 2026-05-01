# Cryostat-card configuration files

This directory holds reusable, hardware-specific configuration files for
individual cryostat cards.  A cryostat-card cfg encodes everything that
depends on the card itself (the resistors loaded onto the card, the PIC
to bias-group mapping, and the bias group to RTM DAC pair mapping) so
that a given card cfg can be referenced from multiple experiment cfgs
that share the same physical card.

A main pysmurf experiment cfg references a card cfg via the optional
top-level `cryostat_card_config_file` key (resolved relative to the
experiment cfg's directory if not absolute):

```jsonc
{
    // ... other experiment cfg keys ...
    "cryostat_card_config_file" : "../cryostat_cards/cc02-06.cfg",
    "Rcable" : 29.1,    // round-trip cable + cold-resistor resistance, ohms
    "R_sh" : 750e-6,
    // bias_line_resistance, high_low_current_ratio, pic_to_bias_group,
    // and bias_group_to_pair are all loaded from the card cfg.
}
```

When loaded, the merge pass in `SmurfConfig.read()`:

1. Reads the card cfg and copies `R_cryostat_card`,
   `high_low_current_ratio`, `pic_to_bias_group`, and `bias_group_to_pair`
   into the experiment cfg if those keys are not already set.  If a key
   is set in both, the experiment cfg wins (explicit override) and a
   notice is printed.
2. Computes `bias_line_resistance = Rcable + R_cryostat_card` if the
   experiment cfg did not set `bias_line_resistance` explicitly.

An explicitly set `bias_line_resistance` always takes precedence
(legacy single-value path), so existing experiment cfgs that bake the
total in directly continue to work unchanged.

## Card-cfg shape

```jsonc
{
    "card_id"                : "cc02-06",  // free-form identifier
    "R_cryostat_card"        : 16505.0,    // round-trip card-only resistance, ohms
    "high_low_current_ratio" : 8.084,
    "pic_to_bias_group"      : { "0": 0, "1": 1, ... },
    "bias_group_to_pair"     : { "0": [1, 2], "1": [3, 4], ... }
}
```

## Why not auto-detect the card?

`CryoCard.get_fw_version()` only distinguishes the major version (C02
vs C04), not which loadout is on the card.  Cards of the same type can
have different on-card resistor values across deployments (e.g. C2-06
vs C2-02 differ).  Auto-detection would silently use the wrong values,
so card cfgs are referenced explicitly.
