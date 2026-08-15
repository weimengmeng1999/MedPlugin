# MedPlugin

Chest X-ray, CT, and MRI specialist tools for [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness). Each tool shells out to a Python script bundled in this repo's `skills/` directory, organized by imaging modality; every script self-manages its own isolated virtual environment and downloads its own model weights on first use.

```
skills/
├── xray/
│   ├── maira2_report.py                    MAIRA-2: report / grounded report / phrase grounding
│   ├── medgemma_anatomy_localization.py    MedGemma 1.5: anatomical structure bounding boxes
│   ├── medgemma_longitudinal.py            MedGemma 1.5: prior-vs-current comparison
│   └── medgemma_report.py                  MedGemma 4B: plain-narrative report
├── ct/
│   ├── medgemma_report.py                  MedGemma 4B: axial-slice-montage report
│   └── totalseg_segmentation.py            TotalSegmentator: organ segmentation
└── mri/
    └── totalseg_segmentation.py            TotalSegmentator: organ segmentation
```

All seven scripts share one venv at `skills/.venv`. Whichever tool runs first creates it and installs the union of every script's dependencies (MAIRA-2's `protobuf`/`sentencepiece`, the MedGemma scripts' `scikit-image`, the CT/MRI scripts' `nibabel`/`simpleitk`/`TotalSegmentator`, and a `transformers` version in `[4.50,4.52)` — the range every script's own tested minimum still tolerates); every other script's own install list is required to match that union exactly, or which script happens to run first would silently decide what's available to the rest. Adding another script under `skills/` means adding its new dependencies to all existing lists too, not just its own.

## Tools

| Tool | Modality | What it does |
|---|---|---|
| `xray_report_maira` | X-ray | Radiology report — MAIRA-2 (plain report, grounded report with bounding boxes, or phrase grounding) |
| `xray_anatomy_localization` | X-ray | Anatomical structure bounding boxes (MedGemma 1.5-4b-it) |
| `xray_longitudinal_comparison` | X-ray | Prior-vs-current interval change (MedGemma 1.5) |
| `xray_report_medgemma` | X-ray | Radiology report — MedGemma 4B (plain narrative text) |
| `ct_report_medgemma` | CT | Radiology report candidate — MedGemma 4B over an axial-slice montage (2D model reasoning over a 3D volume; a complementary candidate, not a substitute for a native 3D CT model) |
| `ct_segmentation_totalseg` | CT | Organ/structure segmentation — TotalSegmentator (whole-body organs by default, or `lung_vessels` for pulmonary vasculature) |
| `mri_segmentation_totalseg` | MRI | Organ/structure segmentation — TotalSegmentator's MR-specific model |

## Requirements

- A deepseek-harness `dsh` installation.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`. Whichever script runs first re-execs itself into a shared venv via `uv venv` + `uv pip install` — if `uv` isn't installed, that bootstrap fails.
- Python 3 on `PATH` to launch each script initially (it re-execs into the shared venv regardless of which Python started it).
- An NVIDIA GPU with CUDA for realistic latency. `gpu: -1` runs on CPU where a tool exposes it, but expect it to be slow.
- A HuggingFace account + `HF_TOKEN` set in the environment (or `huggingface-cli login` already done). MAIRA-2 is a gated model on the Hub — nothing here can automate accepting that license for you. MedGemma is also commonly gated under Google's Health AI Developer Foundations terms; if a MedGemma-backed tool fails on first use with a model-access error, the same `HF_TOKEN` step is almost certainly why.
- For `ct_report_medgemma`: `nibabel` and `simpleitk` (installed into the shared venv, also automatic) convert a `.nii`/`.nii.gz` volume or a DICOM series into the axial-slice montage MedGemma actually sees.
- For the `_totalseg` tools: TotalSegmentator downloads its own nnU-Net model weights on first use, separately from the HuggingFace weights above and without needing `HF_TOKEN`.

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

1. **Venv creation is automatic.** Whichever script runs first re-execs itself into one shared venv, pinning dependency versions (`uv venv` + `uv pip install`). You never run a setup script yourself.
2. **Model weights are automatic** — every script calls HuggingFace's `from_pretrained`, which downloads and caches weights on first use.
3. **Gated-model license acceptance is not automatic and cannot be.** MAIRA-2 needs `HF_TOKEN` after a one-time license click; MedGemma likely does too (see [Requirements](#requirements)).
4. **GPU drivers are a host prerequisite**, same as any other GPU tool.

None of this requires a manual `pip install` or hand-placed checkpoint file — the one-time human steps are `uv` being on `PATH` and accepting each model's Hub license.

## Scope

Two categories of specialist tool are deliberately not in this package:

- **Other model families** (a plain-narrative report generator, a pathology classifier, an alternate phrase-grounding tool, a case-retrieval tool) are technically fine — each would follow the same self-contained-venv pattern as the tools here — but are out of scope for this package, which focuses on MAIRA-2 and MedGemma.
- **Tube/line detection and bone-fracture classification** are excluded for a real technical reason on top of scope: their backing scripts have no isolated-venv bootstrap of their own — they'd import bare `torch`/`transformers` against whatever the ambient Python environment happens to provide, which can silently conflict with a pinned range like MAIRA-2's `transformers>=4.48,<4.52`.

If you want to add tools back in this style, follow the pattern in `index.js`: a `SCRIPTS` entry pointing at a script under `skills/<modality>/`, a matching `defineTool` registration whose `execute` shells out to it and parses its JSON stdout, and the shared-venv bootstrap block from an existing script — with its install list extended to the union described above, and that same extended list copied into every other script too.

## License

MIT — see [LICENSE](./LICENSE).
