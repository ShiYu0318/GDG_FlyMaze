# FlyMaze agent instructions

- Treat `docs/SPEC.md` as the authoritative product and acceptance specification.
- M0, M1, and M2 are complete; preserve their accepted behavior unless a new evidence-backed decision explicitly reopens it.
- Do not promote values marked `[待校準]` or `[待驗證]` without presenting evidence and obtaining the user's decision.
- Never expand the success whitelist beyond the head and the physical left/right forefeet.
- Keep the standard agent observation at exactly seven dimensions.
- Keep `test_straight_unknown` out of demonstrations, training, and hyperparameter selection.
- Use Python 3.12, uv, FlyGym 2.1.1, and the CPU PyTorch index.
- Add regression tests before changing established behavior.
- Keep generated experiment artifacts out of Git unless they are small, explicitly selected review evidence.
