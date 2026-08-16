# MedPlugin

Medical imaging specialist tools for [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness): chest X-ray, CT, MRI, ultrasound, and retinal (fundus) images. Each tool shells out to a Python script bundled in this repo's `skills/` directory, organized by imaging modality; every script self-manages its own isolated virtual environment and downloads its own model weights on first use.

```
skills/
├── _bootstrap.py                            Shared venv bootstrap, imported by every script below
├── biomedparse/
│   ├── __init__.py
│   └── _core.py                             BiomedParse model loading, inference, and overlay rendering — shared by every biomedparse_segmentation.py wrapper below
├── xray/
│   ├── maira2_report.py                     MAIRA-2: report / grounded report / phrase grounding
│   ├── medgemma_anatomy_localization.py     MedGemma 1.5: anatomical structure bounding boxes
│   ├── medgemma_longitudinal.py             MedGemma 1.5: prior-vs-current comparison
│   ├── medgemma_report.py                   MedGemma 4B: plain-narrative report
│   └── biomedparse_segmentation.py          BiomedParse: text-prompted segmentation
├── ct/
│   ├── medgemma_report.py                   MedGemma 4B: axial-slice-montage report
│   ├── totalseg_segmentation.py             TotalSegmentator: organ segmentation
│   └── biomedparse_segmentation.py          BiomedParse: text-prompted segmentation (3D/NIfTI)
├── mri/
│   ├── totalseg_segmentation.py             TotalSegmentator: organ segmentation
│   └── biomedparse_segmentation.py          BiomedParse: text-prompted segmentation (3D/NIfTI)
├── ultrasound/
│   └── biomedparse_segmentation.py          BiomedParse: text-prompted segmentation
└── retinal/
    └── biomedparse_segmentation.py          BiomedParse: text-prompted segmentation
```

Every script above imports `skills/_bootstrap.py` and calls `ensure_venv_and_reexec()` to re-exec itself into one shared venv at `skills/.venv`, pinning dependency versions (`uv venv` + `uv pip install`). Whichever script runs first creates it and installs `_bootstrap.BASE_PACKAGES` — the union of every non-BiomedParse script's dependencies (MAIRA-2's `protobuf`/`sentencepiece`, the MedGemma scripts' `scikit-image`, the CT/MRI scripts' `nibabel`/`simpleitk`/`TotalSegmentator`, and a `transformers` version in `[4.50,4.52)`). Adding a script whose model needs a new base dependency means adding it to `BASE_PACKAGES`, not to one script's own install list — every script installs the same base set, so which one happens to run first can't silently decide what's available to the rest.

BiomedParse needs substantially more — its own cloned model repo, ~15 extra Python packages, and a from-source `detectron2` build (see [Requirements](#requirements)) — so its `biomedparse_segmentation.py` wrappers call `_bootstrap.ensure_extra_packages()` for that group *after* the shared venv exists, gated by a marker file. That keeps the heavy one-time BiomedParse setup scoped to the first time a BiomedParse tool is actually called, instead of landing on whichever script happens to run first — a plain `xray_report_medgemma` call never pays it.

Every `biomedparse_segmentation.py` wrapper is a thin CLI shim (argument parsing + one call into `skills/biomedparse/_core.py`) over the same model and the same inference/overlay code — BiomedParse itself has no modality-specific logic; the modality distinction is which wrapper file `index.js` calls and what its tool description says, not a different model per modality. `_core.py`'s `run_2d()` backs the three image wrappers (xray, ultrasound, retinal); `run_3d()` backs the two NIfTI wrappers (ct, mri), which differ only in whether `is_ct`-specific windowing (`--site`) or multi-channel MRI (`--channel_idx`) applies.

## Tools

| Tool | Modality | What it does |
|---|---|---|
| `xray_report_maira` | X-ray | Radiology report — MAIRA-2 (plain report, grounded report with bounding boxes, or phrase grounding) |
| `xray_anatomy_localization` | X-ray | Anatomical structure bounding boxes (MedGemma 1.5-4b-it) |
| `xray_longitudinal_comparison` | X-ray | Prior-vs-current interval change (MedGemma 1.5) |
| `xray_report_medgemma` | X-ray | Radiology report — MedGemma 4B (plain narrative text) |
| `xray_segmentation_biomedparse` | X-ray | Text-prompted segmentation of any finding/structure — BiomedParse |
| `ct_report_medgemma` | CT | Radiology report candidate — MedGemma 4B over an axial-slice montage (2D model reasoning over a 3D volume; a complementary candidate, not a substitute for a native 3D CT model) |
| `ct_segmentation_totalseg` | CT | Organ/structure segmentation — TotalSegmentator (whole-body organs by default, or `lung_vessels` for pulmonary vasculature) |
| `ct_segmentation_biomedparse` | CT | Text-prompted segmentation of any finding/structure, one slice or the whole volume — BiomedParse |
| `mri_segmentation_totalseg` | MRI | Organ/structure segmentation — TotalSegmentator's MR-specific model |
| `mri_segmentation_biomedparse` | MRI | Text-prompted segmentation of any finding/structure, one slice or the whole volume — BiomedParse |
| `ultrasound_segmentation_biomedparse` | Ultrasound | Text-prompted segmentation of any finding/structure — BiomedParse |
| `retinal_segmentation_biomedparse` | Retinal (fundus) | Text-prompted segmentation of any finding/structure — BiomedParse |

`_totalseg` and `_biomedparse` on the same modality are complementary, not redundant: TotalSegmentator segments a fixed list of named anatomical structures with no text prompt; BiomedParse segments whatever free-text prompt you give it (pathology or anatomy), at the cost of needing you to name what you're looking for.

## Preview images

Every tool also produces one or more 2D preview PNGs alongside its text/file result — a real thing to look at, not just paths. On the X-ray tools this is the input image itself, or (`xray_report_maira`, `xray_anatomy_localization`) that image annotated with the bounding boxes the model found; `xray_longitudinal_comparison` composites prior and current side by side. Neither CT nor MRI input is directly viewable as a 2D image, so `ct_report_medgemma` and the `_totalseg` tools each reuse a 2D projection their own pipeline already produces for a different reason — see the segmentation masks note below.

The BiomedParse tools generate one overlay image per prompt (`xray_segmentation_biomedparse`, `ultrasound_segmentation_biomedparse`, `retinal_segmentation_biomedparse`, and the single-slice mode of `ct_segmentation_biomedparse`/`mri_segmentation_biomedparse`), or one per prompt *per slice* in `all_slices` mode — a chat turn only attaches the first 4 (`MAX_ATTACHED_PREVIEWS` in `index.js`); the full list of generated overlay paths is always in the tool's JSON result regardless of how many got attached.

In every segmentation tool (`_totalseg` and `_biomedparse` alike) the preview shows *that* something was found and roughly *where*; the actual masks stay as files (`segmentation_files` for `_totalseg`, `nifti_masks` for `_biomedparse` in `all_slices` mode), which no chat UI can render inline.

The preview only reaches the conversation when the active model route declares image input; on a text-only route (e.g. the plain DeepSeek chat-completions adapter) the tool call still returns its full text/file result, just without the image attached.

## Install and use

### 1. Install

```sh
dsh plugin --profile <your-profile> add github:weimengmeng1999/MedPlugin
```

Or from a local checkout (useful while developing this repo itself):

```sh
dsh plugin --profile <your-profile> add /path/to/MedPlugin
```

Either form adds `dsh-medplugin` to your profile's `package.json` `dependencies` and `dsh.profile.bundles`, and installs `cordis.patch.yml` (mounting the plugin under id `medplugin`) automatically — see [package and install a plugin](https://deepseek-harness.github.io/deepseek-harness/en/develop/basic/publish) for how bundles and profiles compose. The `github:` form works because this package ships plain `.js`/`.py` with no build step, so there's nothing for a `prepare` script to compile. Restart `dsh` (or your DSH Desktop/web session) afterward so the new bundle loads.

### 2. First run

Nothing needs to be pre-built or downloaded before your first tool call — every stage below happens automatically, the first time it's needed, inside whichever tool call triggers it. What you'll see on `stderr` at each stage:

| Stage | Trigger | What happens | You'll see |
|---|---|---|---|
| Shared venv | Any tool's first call, ever | `uv venv` creates `skills/.venv`, pins `torch` to the ambient version, installs the base dependency set | `[<tool-tag>] Creating venv ...` then `... Venv ready.` |
| Model weights | A MAIRA-2/MedGemma tool's first call | Downloads from HuggingFace via `from_pretrained` (gated — see [Requirements](#requirements)) | Standard `huggingface_hub` download progress |
| BiomedParse repo + deps | Any `_segmentation_biomedparse` tool's first call | Clones `microsoft/BiomedParse`'s `inference` branch, installs ~15 extra packages, builds `detectron2` from source (~1-2 min) | `[biomedparse-<modality>] Cloning BiomedParse inference branch ...`, `... Installing biomedparse-deps dependencies ...`, `... Installing biomedparse-detectron2 dependencies ...` |
| BiomedParse weights | Same first call, right after | Downloads `biomedparse_v1.pt` (~1.7GB, ungated) from HuggingFace | `[biomedparse-<modality>] Downloading biomedparse_v1.pt from microsoft/BiomedParse ...` |

The shared-venv stage only happens once total, regardless of which tool triggers it. The BiomedParse stage only happens once total across all five `_biomedparse` tools (the marker files it leaves in `skills/.venv/` are what make the second `_biomedparse` call, and every call after, skip straight to inference). Run `dsh-medplugin doctor` at any point to see exactly which of these stages has completed (see [Troubleshooting](#troubleshooting)).

### 3. Usage examples

These are real tool calls, not a hypothetical — ask your agent something like the phrasing below and it picks the matching tool itself.

**Chest X-ray report:**
> "Generate a radiology report for this chest X-ray: `/path/to/chest_xray.png`"

calls `xray_report_medgemma` (or `xray_report_maira` for a grounded report with bounding boxes), and returns findings text plus an annotated preview image when your active model route supports image input.

**Find something specific in an image, by name, on any of the five modalities:**
> "Does this chest X-ray show consolidation or pleural effusion?"
> "Segment the gallstone in this ultrasound image."
> "Are there any microaneurysms in this fundus photo?"

calls `xray_segmentation_biomedparse` / `ultrasound_segmentation_biomedparse` / `retinal_segmentation_biomedparse` with your finding as the `prompts` argument, and returns a coverage percentage plus an overlay per prompt — BiomedParse takes any free-text finding or anatomical structure, not a fixed list.

**Whole-body organ segmentation on a CT or MRI volume:**
> "Segment the liver and kidneys in this CT scan: `/path/to/scan.nii.gz`"

calls `ct_segmentation_totalseg` for the fixed anatomical-structure list, or `ct_segmentation_biomedparse` (with `site: "abdomen"`) if you'd rather name a pathology than an organ — writes one NIfTI mask per structure/prompt plus a preview showing where they landed.

**Prior-vs-current comparison:**
> "Compare this current chest X-ray to the prior one and describe interval change."

calls `xray_longitudinal_comparison` with both images, and returns Improved/Stable/Worsened findings across the documented pathology categories.

### Troubleshooting

`dsh-medplugin doctor` checks this machine's setup without calling any model — whether `uv`/`git`/`python3` are on `PATH`, whether the shared venv exists, and how far BiomedParse's one-time setup has progressed. Installing this plugin already installs the CLI as a dependency, so run it from your profile directory:

```sh
cd ~/.dsh/profiles/<your-profile>
./node_modules/.bin/dsh-medplugin doctor
```

or, against a local checkout of this repo directly:

```sh
node /path/to/MedPlugin/lib/doctor-cli.js
```

```
skills/: /path/to/MedPlugin/skills
✓ uv — uv 0.10.7
✓ git — git version 2.34.1
✓ python3 — Python 3.12.12

✓ shared venv
    torch pin: torch==2.10.0
    torch 2.10.0+cu128, CUDA available: true

✓ BiomedParse repo cloned
✓ BiomedParse extra dependencies installed
✓ detectron2 built
✓ BiomedParse weights downloaded — 1.7GB
```

A `✗` line names exactly what's missing and why it matters (e.g. `git not found on PATH (required to clone the BiomedParse model repo)`). An unchecked BiomedParse line is not itself an error — that stage only runs on a `_biomedparse` tool's first call (see [First run](#2-first-run)) — but is the first place to look if such a call fails. Pass `--skills-dir <path>` if your profile overrides the plugin's default `skillsDir` (see [Configure](#configure)).

## Requirements

- A deepseek-harness `dsh` installation.
- [`uv`](https://docs.astral.sh/uv/) on `PATH`. Whichever script runs first re-execs itself into a shared venv via `uv venv` + `uv pip install` — if `uv` isn't installed, that bootstrap fails.
- Python 3 on `PATH` to launch each script initially (it re-execs into the shared venv regardless of which Python started it).
- `git` on `PATH` — the BiomedParse tools clone [microsoft/BiomedParse](https://github.com/microsoft/BiomedParse)'s `inference` branch into `skills/biomedparse/BiomedParse/` on first use.
- A working C/C++ build toolchain — the BiomedParse tools build [detectron2](https://github.com/facebookresearch/detectron2) from source (`--no-build-isolation`, against the venv's already-installed `torch`) on first use, which takes roughly a minute or two.
- An NVIDIA GPU with CUDA for realistic latency. `gpu: -1` runs on CPU where a tool exposes it, but expect it to be slow.
- A HuggingFace account + `HF_TOKEN` set in the environment (or `huggingface-cli login` already done) for MAIRA-2 and MedGemma. MAIRA-2 is a gated model on the Hub — nothing here can automate accepting that license for you. MedGemma is also commonly gated under Google's Health AI Developer Foundations terms; if a MedGemma-backed tool fails on first use with a model-access error, the same `HF_TOKEN` step is almost certainly why. **BiomedParse's own weights are not gated** — no token is required for any `_biomedparse` tool.
- For `ct_report_medgemma`: `nibabel` and `simpleitk` (installed into the shared venv, also automatic) convert a `.nii`/`.nii.gz` volume or a DICOM series into the axial-slice montage MedGemma actually sees.
- For the `_totalseg` tools: TotalSegmentator downloads its own nnU-Net model weights on first use, separately from the HuggingFace weights above and without needing `HF_TOKEN`.

## Configure

Nothing is required — the plugin defaults to the `skills/` directory shipped inside this package. Override only if you want to point at a different copy of these scripts, in your profile's `cordis.patch.yml`:

```yaml
- upsert:
    - id: medplugin
      config:
        skillsDir: /path/to/other/skills   # optional, default: this package's own skills/
        # pythonBin: python3               # optional, default "python3"
        # timeoutMs: 1800000                # optional, default 30 minutes
```

## Scope

Two categories of specialist tool are deliberately not in this package:

- **Other model families** beyond MAIRA-2, MedGemma, TotalSegmentator, and BiomedParse are technically fine — each would follow the same self-contained-venv pattern as the tools here — but are out of scope for this package.
- **Tube/line detection and bone-fracture classification** are excluded for a real technical reason on top of scope: their backing scripts have no isolated-venv bootstrap of their own — they'd import bare `torch`/`transformers` against whatever the ambient Python environment happens to provide, which can silently conflict with a pinned range like MAIRA-2's `transformers>=4.48,<4.52`.

If you want to add tools back in this style, follow the pattern in `index.js`: a `SCRIPTS` entry pointing at a script under `skills/<modality>/`, a matching `defineTool` registration whose `execute` shells out to it and parses its JSON stdout, and `skills/_bootstrap.py`'s bootstrap at the top of the script. A model that only needs `_bootstrap.BASE_PACKAGES` calls just `ensure_venv_and_reexec()`; one with heavier or model-specific dependencies (BiomedParse's `detectron2` build is the existing example) should call `ensure_extra_packages()` for that group after the reexec, so scripts that don't need it never pay its cost.

## License

MIT — see [LICENSE](./LICENSE).
