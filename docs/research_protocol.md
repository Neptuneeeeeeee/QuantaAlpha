# Research Protocol Preflight

QuantaAlpha uses different date roles during factor discovery and during the
independent backtest. The optional `ResearchProtocol` makes those roles
explicit and lets researchers detect contradictory configuration before an
expensive run.

This is the first slice of the evaluation/provenance work proposed in
[issue #37](https://github.com/QuantaAlpha/QuantaAlpha/issues/37). It is a
**configuration preflight**, not a data sandbox.

## Current role mapping

For the repository's current templates, preflight resolves:

| Stage | Role | Period | Existing Qlib/config name |
| --- | --- | --- | --- |
| Discovery | fit | 2016-01-01 to 2019-12-31 | `train` |
| Discovery | tune | 2020-01-01 to 2020-12-31 | `valid` |
| Discovery | adaptive selection | 2021-01-01 to 2021-12-31 | `test` |
| Final | fit/refit | 2016-01-01 to 2020-12-31 | `train` |
| Final | tune | 2021-01-01 to 2021-12-31 | `valid` |
| Final | evaluation | 2022-01-01 to 2025-12-26 | `test` |

The repeated word `test` in Qlib configuration therefore does not represent
the same research role in both stages.

## Inspect existing configuration

No protocol file is required to inspect the effective roles:

```bash
quantaalpha protocol inspect
```

For machine-readable output:

```bash
quantaalpha protocol inspect --format json
```

The command reads the two mining templates and `configs/backtest.yaml` by
default. It does not initialize Qlib, load market data, call an LLM, or make a
network request.

## Validate an explicit protocol

Start from `configs/research_protocol.example.yaml` and pass it explicitly:

```bash
quantaalpha protocol inspect \
  --protocol configs/research_protocol.example.yaml
```

When a protocol is supplied, the declared universe, stage ranges, and supported
label semantics must match the effective configurations. A disagreement is an
error; the preflight does not silently rewrite existing YAML.

The protocol is opt-in. Existing `mine`, `backtest`, `health_check`, and
`collect_info` behavior remains unchanged when no protocol command is used.

## Preview artifacts

To write a canonical manifest and report:

```bash
quantaalpha protocol preview \
  --protocol configs/research_protocol.example.yaml \
  --output-dir data/results/protocol_preview
```

Preview writes only:

- `research_protocol_manifest.json`
- `research_protocol_report.json`

The manifest is built from a whitelist of public protocol fields and has a
stable SHA-256 identity. Formatting differences and quoted versus YAML-native
dates do not change the canonical representation.

## Label availability

Protocol v1 recognizes exactly the current label:

```text
Ref($close, -2) / Ref($close, -1) - 1
```

Its declared availability semantics are:

- entry offset: 1 trading session
- endpoint offset: 2 trading sessions
- return duration: 1 trading session

Unknown expressions are reported as unsupported. Preflight deliberately does
not infer a generic horizon from arbitrary expressions.

Session-aware purge/embargo rows require a versioned trading calendar.
`calendar_id: null` means the boundary cannot yet be resolved; preflight does
not substitute calendar days for exchange sessions.

## What this does not guarantee

A successful preflight means the **configuration contract** is internally
consistent. It does **not** establish any of the following:

- that factor code cannot read observations after the adaptive-selection cutoff;
- that every factor operator is causal;
- that caches were computed from the same market-data snapshot and cutoff;
- that the final-evaluation period has never been observed;
- that generated Python is sandboxed;
- that a reported metric is independent of adaptive research decisions.

Those guarantees require the follow-on work described in issue #37: bounded
as-of data views, temporal operator tests, context-aware cache/artifact
identities, typed evaluation reports, and final-evaluation controls.
