# lite-llm-inference

Inference runtime crate for Lite LLM (`SPEC-041` to `SPEC-050`).

## Scope
Implements deterministic inference primitives:

- TierSet selection engine and budget solver
- token routing execution and expert packing/dispatch
- prefetch planning and KV-cache behavior
- streaming session runtime with replayable prefixes
- cost-adaptive routing and telemetry
- multi-tenant isolation controls

## Modules
- `src/tierset_selection.rs`
- `src/pipeline.rs`
- `src/prefetch.rs`
- `src/kv_cache.rs`
- `src/streaming.rs`
- `src/cost_adaptive.rs`
- `src/telemetry.rs`
- `src/tenant.rs`
- `src/types.rs`
- `src/error.rs`

## Build and Test
```bash
cargo fmt
cargo test
```

## Documentation
- System docs: `../lite-llm-docs/README.md`
- API docs: `../lite-llm-docs/api/mode-entrypoints.md`

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
