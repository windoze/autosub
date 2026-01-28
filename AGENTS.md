# Repository Guidelines

## Project Structure & Module Organization

- `src/` — main `autosub` crate: CLI entrypoint (`main.rs`) plus modules like `audio.rs`, `srt.rs`, `translate.rs`, and `config.rs`.
- `autosub-asr/` — ASR/Whisper engine (Candle-based) and audio/VAD utilities. Precomputed mel filter assets live in `autosub-asr/src/*.bytes`.
- `scripts/` — maintenance tools (for example, `generate_melfilters.py` to regenerate mel filters).
- `.github/workflows/` — CI and release pipelines.
- `test/` — local fixtures (gitignored); keep large media files out of Git history.

## Build, Test, and Development Commands

- `cargo build` — compile a debug build.
- `cargo build --release` — optimized build (binary at `target/release/autosub`).
- `cargo run -- <INPUT> -v` — run locally (example: `cargo run -- ./video.mp4 -v`).
- `cargo test` — run unit tests.
- `cargo fmt` — format with rustfmt (required before PRs).
- `cargo clippy --all-targets --all-features -- -D warnings` — lint with Clippy.

Feature flags (when needed):
- `--features cuda` — CUDA GPU acceleration (Metal is automatically enabled on macOS).
- `--features ffmpeg-static` / `--features ffmpeg-build` — produce self-contained binaries without runtime FFmpeg.

## Coding Style & Naming Conventions

- Rust 2021, standard rustfmt formatting (4-space indentation).
- Use idiomatic Rust naming: `snake_case` (functions/modules), `CamelCase` (types), `SCREAMING_SNAKE_CASE` (constants).
- Prefer `thiserror` for reusable error types and `anyhow` at CLI boundaries.

## Testing Guidelines

- Keep tests deterministic and offline: avoid calling external LLM providers or downloading large models in tests.
- Add unit tests close to the logic (`#[cfg(test)]` in the same module) and keep them fast.
- For end-to-end checks, use a short local clip and verify emitted `.srt` formatting and timing.

## Commit & Pull Request Guidelines

- Commit messages in this repo are typically imperative and concise (e.g., “Add …”, “Fix …”, “Update …”); keep the subject ≤72 characters.
- PRs should include: what changed, how to test (exact commands), and platform notes when touching FFmpeg/GPU paths (macOS/Linux/Windows).
- Do not commit secrets or local config. Use environment variables (e.g., `AUTOSUB_LLM_API_KEY`, `OPENAI_API_KEY`, `HF_ENDPOINT`) and keep `.envrc`/fixtures untracked.
