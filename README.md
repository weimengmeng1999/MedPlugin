# MedPlugin

Chest X-ray specialist tools for [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness). Each tool shells out to a Python script bundled in this repo's `skills/` directory; every script self-manages its own isolated virtual environment and downloads its own model weights on first use.

## Tools

Scoped to two model families — MAIRA-2 and MedGemma 1.5:

| Tool | What it does |
|---|---|
| `xray_report_maira` | Radiology report — MAIRA-2 (plain report, grounded report with bounding boxes, or phrase grounding) |
| `xray_anatomy_localization` | Anatomical structure bounding boxes (MedGemma 1.5-4b-it) |
| `xray_longitudinal_comparison` | Prior-vs-current interval change (MedGemma 1.5) |

## Requirements

- A deepseek-harness `dsh` installation.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`. Every script re-execs itself into its own isolated `.venv` on first run via `uv venv` + `uv pip install` — if `uv` isn't installed, that bootstrap fails.
- Python 3 on `PATH` to launch each script initially (it re-execs into its own venv regardless of which Python started it).
- An NVIDIA GPU with CUDA for realistic latency. `gpu: -1` runs on CPU where a tool exposes it, but expect it to be slow.
- A HuggingFace account + `HF_TOKEN` set in the environment (or `huggingface-cli login` already done). MAIRA-2 is a gated model on the Hub — nothing here can automate accepting that license for you. MedGemma is also commonly gated under Google's Health AI Developer Foundations terms; if `xray_anatomy_localization` / `xray_longitudinal_comparison` fail on first use with a model-access error, the same `HF_TOKEN` step is almost certainly why.

## Install

```sh
dsh plugin --profile <your-profile> add github:weimengmeng1999/MedPlugin
```

Or from a local checkout:

```sh
dsh plugin --profile <your-profile> add /path/to/MedPlugin
```

See deepseek-harness's [package and install a plugin](https://deepseek-harness.github.io/deepseek-harness/en/develop/basic/publish) tutorial for how bundles and profiles compose, and note the git-install caveat there: a `github:` install only works because this package ships plain `.js` with no build step — there's nothing for a `prepare` script to compile.

## Configure

Nothing is required — the plugin defaults to the `skills/` directory shipped inside this package. Override only if you want to point at a different copy of these scripts, in your profile's `cordis.patch.yml`:

```yaml
- upsert:
    - id: xray-report-generation
      config:
        skillsDir: /path/to/other/skills   # optional, default: this package's own skills/
        # pythonBin: python3               # optional, default "python3"
        # timeoutMs: 1800000                # optional, default 30 minutes
```

## Is setup automated?

Mostly, yes — but there are real, non-automatable exceptions worth knowing about before you rely on this:

1. **Venv creation is automatic.** All three scripts re-exec themselves into their own isolated `.venv` on first invocation, pinning their own dependency versions (`uv venv` + `uv pip install`). You never run a setup script yourself.
2. **Model weights are automatic** — all three call HuggingFace's `from_pretrained`, which downloads and caches weights on first use.
3. **Gated-model license acceptance is not automatic and cannot be.** MAIRA-2 needs `HF_TOKEN` after a one-time license click; MedGemma likely does too (see [Requirements](#requirements)).
4. **GPU drivers are a host prerequisite**, same as any other GPU tool.

None of this requires a manual `pip install` or hand-placed checkpoint file — the one-time human steps are `uv` being on `PATH` and accepting each model's Hub license.

## Scope

Two categories of chest X-ray specialist tool are deliberately not in this package:

- **Other model families** (a plain-narrative report generator, a pathology classifier, an alternate phrase-grounding tool, a case-retrieval tool) are technically fine — each would follow the same self-contained-venv pattern as the three tools here — but are out of scope for this package, which focuses on MAIRA-2 and MedGemma 1.5.
- **Tube/line detection and bone-fracture classification** are excluded for a real technical reason on top of scope: their backing scripts have no isolated-venv bootstrap of their own — they'd import bare `torch`/`transformers` against whatever the ambient Python environment happens to provide, which can silently conflict with MAIRA-2's pinned `transformers>=4.48,<4.52`.

If you want to add tools back in this style, follow the pattern in `index.js`: a `SCRIPTS` entry pointing at a script under `skills/`, and a matching `defineTool` registration whose `execute` shells out to it and parses its JSON stdout.

## License

MIT — see [LICENSE](./LICENSE).
