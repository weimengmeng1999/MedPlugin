#!/usr/bin/env python3
"""
Meng Wei

CT report-generation candidate with MedGemma.

MedGemma is a 2-D image-text-to-text model, so this script converts a CT volume
into a fixed axial-slice montage. Treat this as a complementary candidate for a
selector, not as a replacement for Merlin's native 3-D CT model.
"""

import os
import subprocess
import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
_VENV_DIR = _SKILL_DIR.parent / ".venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"


def _ensure_venv_and_reexec() -> None:
    if sys.executable == str(_VENV_PYTHON):
        return

    if not _VENV_PYTHON.exists():
        print("[medgemma-ct] Creating isolated venv ...", file=sys.stderr)
        subprocess.check_call(
            ["uv", "venv", "--system-site-packages", "--python", sys.executable, str(_VENV_DIR)],
        )

        import importlib.metadata

        try:
            torch_pin = f"torch=={importlib.metadata.version('torch')}"
        except importlib.metadata.PackageNotFoundError:
            torch_pin = None
        constraints_file = _SKILL_DIR.parent / ".uv-constraints.txt"
        constraints_file.write_text(f"{torch_pin}\n" if torch_pin else "")

        subprocess.check_call(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(_VENV_PYTHON),
                "--constraint",
                str(constraints_file),
                "transformers>=4.50.0,<4.52",
                "accelerate",
                "pillow",
                "protobuf",
                "sentencepiece",
                "numpy",
                "scikit-image",
                "nibabel",
                "simpleitk",
            ],
        )
        print("[medgemma-ct] Venv ready.", file=sys.stderr)

    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)


_ensure_venv_and_reexec()

import argparse
import json
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw


SYSTEM_PROMPT = "You are an expert radiologist."


def dicom_to_nifti(dicom_dir: Path) -> str:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(names)
    image = reader.Execute()
    tmp_dir = tempfile.mkdtemp(prefix="medgemma_ct_dicom_")
    out = str(Path(tmp_dir) / "ct_volume.nii.gz")
    sitk.WriteImage(image, out)
    return out


def resolve_nifti(path_text: str) -> str:
    path = Path(path_text).expanduser().resolve()
    if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return str(path)
    if path.is_dir():
        return dicom_to_nifti(path)
    raise ValueError("Input must be a .nii/.nii.gz file or a DICOM series directory")


def window_slice(slice_2d: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(slice_2d, low, high)
    return ((clipped - low) / (high - low) * 255.0).astype(np.uint8)


def make_axial_montage(nifti_path: str, n_slices: int = 16, tile_size: int = 224) -> Image.Image:
    import nibabel as nib

    img = nib.load(nifti_path)
    volume = np.asanyarray(img.dataobj)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D CT volume, got shape {volume.shape}")

    z_count = volume.shape[2]
    start = max(0, int(z_count * 0.08))
    stop = min(z_count - 1, int(z_count * 0.92))
    indices = np.linspace(start, stop, n_slices).astype(int)

    tiles = []
    for idx in indices:
        arr = window_slice(volume[:, :, idx])
        tile = Image.fromarray(np.rot90(arr)).convert("RGB")
        tile = tile.resize((tile_size, tile_size), Image.BILINEAR)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 58, 18), fill=(0, 0, 0))
        draw.text((4, 3), f"z={idx}", fill=(255, 255, 255))
        tiles.append(tile)

    cols = int(np.ceil(np.sqrt(n_slices)))
    rows = int(np.ceil(n_slices / cols))
    montage = Image.new("RGB", (cols * tile_size, rows * tile_size), (0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % cols) * tile_size
        y = (i // cols) * tile_size
        montage.paste(tile, (x, y))
    return montage


def load_pipeline(gpu: int, model_id: str):
    import torch
    from transformers import pipeline

    device_map = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    print(f"[medgemma-ct] Loading {model_id} on {device_map}", file=sys.stderr)
    return pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def run_report(pipe, image: Image.Image, indication: str | None, max_new_tokens: int) -> str:
    prompt = (
        "Generate the findings section of an abdominal CT radiology report from "
        "this axial CT slice montage. Be concise, organize by organ system, and "
        "avoid findings that are not visible."
    )
    if indication:
        prompt += f" Clinical indication: {indication}"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    content = output[0]["generated_text"][-1]["content"]
    if isinstance(content, list):
        response = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        response = str(content)
    if "<unused95>" in response:
        response = response.split("<unused95>", 1)[1].lstrip()
    return response.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="MedGemma CT montage report candidate")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--study_id", default=None)
    parser.add_argument("--indication", default=None)
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--n_slices", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--montage_output", default=None)
    args = parser.parse_args()

    t0 = time.time()
    try:
        nifti_path = resolve_nifti(args.input)
        montage = make_axial_montage(nifti_path, args.n_slices)
        if args.montage_output:
            Path(args.montage_output).parent.mkdir(parents=True, exist_ok=True)
            montage.save(args.montage_output)
        pipe = load_pipeline(args.gpu, args.model)
        report_text = run_report(pipe, montage, args.indication, args.max_new_tokens)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        sys.exit(1)

    result = {
        "status": "success",
        "study_id": args.study_id,
        "model": args.model,
        "mode": "ct_montage_report",
        "image_path": args.input,
        "resolved_nifti_path": nifti_path,
        "report_text": report_text,
        "elapsed_seconds": round(time.time() - t0, 2),
        "research_only": True,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
