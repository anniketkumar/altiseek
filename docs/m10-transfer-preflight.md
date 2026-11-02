# M10 transfer B preflight (protocol v2 `31b45c0d400dbc06`, 2026-11-02)

These are B integration checks with the refit sensor `caba50dfc568562d`, flown
before any T run. They are not held-out results. The run root is
`/home/aniket/altiseek-m10/transfers/31b45c0d400dbc06/preflight/`.

| Policy | Views | Actions | Levels visited | Sim time (s) | Wall (s) | Found | False decl. | Max goal error (m) |
|---|---|---|---|---|---|---|---|---|
| B5 | 2 | 1 | L5 | 21.4 | 151 | 1/1 | 0 | 0.264 |
| PPO 17 | 4 | 3 (descend, move, climb) | L5, L4 | 41.6 | 170 | 1/1 | 0 | 0.371 |
| PPO 29 | 3 | 2 (descend, move) | L5, L4 | 27.4 | 140 | 1/1 | 0 | 0.318 |
| PPO 43 | 16 | 15 | L5, L4 | 223.4 | 335 | 1/1 | 1 | 0.379 |

- Every case passed its identity and completion checks: correct protocol,
  checkpoint, sensor case and paired seeds, measured 500 s energy, normal
  launch, an ended episode, and confirmed process cleanup.
- No capture was rejected, so the D56 recapture path was not exercised.
- Wall time ≈ 115–130 s of fixed overhead plus simulated search time,
  consistent with real-time Gazebo.
- v1 (`32c7970163d73479`, world `003b0c54f581c65b`) passed too, but every
  episode ended on the first view and no policy acted. It was superseded
  before any T run; see `docs/m10-transfer-protocol.md`.
