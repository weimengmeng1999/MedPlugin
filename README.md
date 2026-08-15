# MedPlugin

Chest X-ray specialist tools for [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), backed by the model-inference scripts in [MedOmni](https://github.com/weimengmeng1999/MedOmni)'s `skills_scripts/`. Each tool shells out to a standalone Python script that self-manages its own isolated virtual environment and model weights — this package is a thin dsh-tool wrapper, not a copy of that code.

## Tools

Scoped to two model families — MAIRA-2 and MedGemma 1.5:

| Tool | What it does |
|---|---|
| `xray_report_maira` | Radiology report — MAIRA-2 (plain report, grounded report with bounding boxes, or phrase grounding) |
| `xray_anatomy_localization` | Anatomical structure bounding boxes (MedGemma 1.5-4b-it) |
| `xray_longitudinal_comparison` | Prior-vs-current interval change (MedGemma 1.5) |

MedOmni's `xray_team.py` exposes more specialist tools than this — CheXagent, LLaVA-Rad, DenseNet121 classification, RadZero grounding, CARZero case retrieval, plus CarinaNet detection and SigLIP2 fracture classification. All are left out of this package by scope, not because they're broken: see [Scope](#scope) for why.

## Requirements

- A deepseek-harness `dsh` installation.
- A local checkout of [MedOmni](https://github.com/weimengmeng1999/MedOmni) — this plugin calls scripts under its `skills_scripts/` directory; it does not vendor them.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`. Every script re-execs itself into its own isolated `.venv*` on first run via `uv venv`/`uv sync` — if `uv` isn't installed, that bootstrap fails.
- Python 3 on `PATH` to launch each script initially (it re-execs into its own venv regardless of which Python started it).
- An NVIDIA GPU with CUDA for realistic latency. `gpu: -1` runs on CPU where a tool exposes it, but expect it to be slow.
- A HuggingFace account + `HF_TOKEN` set in the environment (or `huggingface-cli login` already done) for `xray_report_maira` — MAIRA-2 is a gated model on the Hub. Nothing here can automate accepting that license for you. MedGemma is also commonly gated on the Hub under Google's Health AI Developer Foundations terms; if `xray_anatomy_localization` / `xray_longitudinal_comparison` fail on first use with a model-access error, the same `HF_TOKEN` step is almost certainly why.

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

Point the plugin at your MedOmni checkout's `skills_scripts/` directory — required, since there's no portable default. Either set it in your profile's `cordis.patch.yml`:

```yaml
- upsert:
    - id: xray-report-generation
      config:
        baseDir: /path/to/MedOmni/skills_scripts
        # pythonBin: python3       # optional, default "python3"
        # timeoutMs: 1800000       # optional, default 30 minutes
```

or export `MEDPLUGIN_SKILLS_DIR=/path/to/MedOmni/skills_scripts` in the environment `dsh` runs in. The plugin fails loud at load time (a clear config error, not a confusing subprocess failure) if neither is set or the directory doesn't exist.

## Is setup automated?

Mostly, yes — but there are real, non-automatable exceptions worth knowing about before you rely on this:

1. **Venv creation is automatic.** All three scripts re-exec themselves into their own isolated `.venv*` on first invocation, pinning their own dependency versions (`uv venv` + `uv sync`). You never run a setup script yourself.
2. **Model weights are automatic** — all three call HuggingFace's `from_pretrained`, which downloads and caches weights on first use.
3. **Gated-model license acceptance is not automatic and cannot be.** MAIRA-2 needs `HF_TOKEN` after a one-time license click; MedGemma likely does too (see [Requirements](#requirements)).
4. **GPU drivers are a host prerequisite**, same as any other GPU tool.

None of this requires a manual `pip install` or hand-placed checkpoint file — the one-time human steps are `uv` being on `PATH` and accepting each model's Hub license.

## Scope

MedOmni's `xray_team.py` exposes more specialist tools than this package ships. Two categories were left out, for different reasons:

- **CheXagent, LLaVA-Rad, DenseNet121 classification, RadZero grounding, and CARZero case retrieval** are all technically fine — each self-manages its own isolated venv the same way MAIRA-2 and MedGemma do — but are out of scope for this package, which focuses on MAIRA-2 and MedGemma 1.5.
- **CarinaNet detection (`detection_tool`) and SigLIP2 fracture classification (`bone_tool`)** are excluded for a real technical reason on top of scope: their backing scripts (`run_xray_detection.py`, `run_xray_bone_classification.py`) have no isolated-venv bootstrap — they import bare `torch`/`transformers` against whatever the ambient Python environment happens to provide, which can silently conflict with MAIRA-2's pinned `transformers>=4.48,<4.52`.

If you want any of these back, they follow the same pattern as the three tools here — add a `SCRIPTS` entry pointing at the script under your MedOmni checkout and a matching `defineTool` registration in `index.js`.

## License

MIT — see [LICENSE](./LICENSE).
