# MedPlugin

Chest X-ray specialist tools for [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), backed by the model-inference scripts in [MedOmni](https://github.com/weimengmeng1999/MedOmni)'s `skills_scripts/`. Each tool shells out to a standalone Python script that self-manages its own isolated virtual environment and model weights — this package is a thin dsh-tool wrapper, not a copy of that code.

## Tools

| Tool | What it does |
|---|---|
| `xray_report_chexagent` | Radiology report — CheXagent-8b (fast, plain narrative text) |
| `xray_report_llava_rad` | Radiology report — LLaVA-Rad, optionally reasoning over draft reports from other models |
| `xray_report_maira` | Radiology report — MAIRA-2 (plain report, grounded report with bounding boxes, or phrase grounding) |
| `xray_classification` | 18-class pathology classification (DenseNet121) |
| `xray_anatomy_localization` | Anatomical structure bounding boxes (MedGemma 1.5-4b-it) |
| `xray_longitudinal_comparison` | Prior-vs-current interval change (MedGemma 1.5) |
| `xray_radzero_grounding` | Zero-shot phrase grounding with a presence score (RadZero) |
| `xray_case_retrieval` | Similar-case retrieval against a pre-built case index (CARZero) |

All eight mirror the tools `xray_team.py`'s `build_xray_specialist_tools()` exposes in MedOmni, minus two intentionally left out — see [Why detection and bone fracture aren't here](#why-detection-and-bone-fracture-arent-here).

## Requirements

- A deepseek-harness `dsh` installation.
- A local checkout of [MedOmni](https://github.com/weimengmeng1999/MedOmni) — this plugin calls scripts under its `skills_scripts/` directory; it does not vendor them.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`. Every script re-execs itself into its own isolated `.venv*` on first run via `uv venv`/`uv sync` — if `uv` isn't installed, that bootstrap fails.
- Python 3 on `PATH` to launch each script initially (it re-execs into its own venv regardless of which Python started it).
- An NVIDIA GPU with CUDA for realistic latency. `gpu: -1` runs on CPU where a tool exposes it, but expect it to be slow.
- A HuggingFace account + `HF_TOKEN` set in the environment (or `huggingface-cli login` already done) for `xray_report_maira` — MAIRA-2 is a gated model on the Hub. Nothing here can automate accepting that license for you.

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

1. **Venv creation is automatic.** Every script kept here re-execs itself into its own isolated `.venv*` on first invocation, pinning its own dependency versions (`uv venv` + `uv sync`). You never run a setup script yourself.
2. **Model weights are automatic for the report/classification/localization tools** — they call HuggingFace's `from_pretrained`, which downloads and caches weights on first use.
3. **`xray_case_retrieval` is also automatic, but from Google Drive, not HuggingFace** — its first call fetches ~6.6GB of CARZero checkpoints via `gdown`. This can be slow, and Drive enforces a download quota on large files that occasionally blocks it entirely; a failure here is Drive rate-limiting, not a bug in this plugin.
4. **`xray_report_maira` needs a human to accept MAIRA-2's gated-model license and provide `HF_TOKEN` once.** No amount of automation gets around a license click.
5. **GPU drivers are a host prerequisite**, same as any other GPU tool.

None of this requires a manual `pip install` or hand-placed checkpoint file — the one-time human steps are `uv` being on `PATH` and the MAIRA-2 token.

## Why detection and bone fracture aren't here

MedOmni's `xray_team.py` exposes two more specialist tools this package deliberately omits: `detection_tool` (CarinaNet, ETT/carina position) and `bone_tool` (SigLIP2 fracture detection). Unlike every tool above, their backing scripts (`run_xray_detection.py`, `run_xray_bone_classification.py`) have no isolated-venv bootstrap — they import bare `torch`/`transformers` against whatever the ambient Python environment happens to provide. That's a real conflict risk: MAIRA-2 alone requires `transformers>=4.48,<4.52`, a narrower range than a shared environment is likely to have. Rather than ship two tools whose reliability depends on the specific ambient environment they happen to run in, this package keeps only the eight that fully own their dependencies.

## License

MIT — see [LICENSE](./LICENSE).
