# Architecture of the `circuit` module — see `CIRCUIT_DEVELOPER_MANUAL.md`

This document has been folded into the comprehensive developer's
manual at [`CIRCUIT_DEVELOPER_MANUAL.md`](./CIRCUIT_DEVELOPER_MANUAL.md).
Maintaining two near-overlapping documents created drift risk; the
manual is now the single source of truth for the module's design and
internals.

For the content this stub used to hold:

- **Quick orientation** — see manual §1 (module map) and §2 (the
  two-stage `SymbolicCircuit → Circuit` pipeline).
- **Class hierarchy and mixin composition** — see manual §3.
- **End-to-end flow from YAML to eigenvalues** — see manual §4
  (graph layer), §5 (symbolic layer), §6 (numerical layer), §7
  (hierarchical diagonalisation), §8 (operator generation).
- **Public API surface** — see manual §13.
- **Where to look for what** lookup table — see manual §18.

For verification of behavioural preservation across refactors, see
[`VERIFY_REFACTOR.md`](./VERIFY_REFACTOR.md).
