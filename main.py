import subprocess
import tempfile
import os
import shutil
import pathlib
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Disposition"],
)

BASE_DIR = pathlib.Path(__file__).parent
SCRIPT   = BASE_DIR / "inference_realesrgan.py"
WEIGHTS  = BASE_DIR / "weights"
MAX_BYTES = 20 * 1024 * 1024

@app.get("/health")
def health():
    return {
        "status": "ok",
        "x4_ready": (WEIGHTS / "RealESRGAN_x4plus.pth").exists(),
        "x2_ready": (WEIGHTS / "RealESRGAN_x2plus.pth").exists(),
    }

@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    scale: int = Query(default=4, ge=2, le=4),
):
    if scale not in (2, 4):
        raise HTTPException(400, detail="scale must be 2 or 4")

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, detail="Unsupported type. Use JPEG, PNG, or WebP.")

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, detail="File exceeds 20 MB limit.")

    model_name = "RealESRGAN_x4plus" if scale == 4 else "RealESRGAN_x2plus"
    model_path = WEIGHTS / f"{model_name}.pth"
    if not model_path.exists():
        raise HTTPException(503, detail=f"Model weights missing: {model_path.name}")

    fname  = file.filename or "input.png"
    ext    = fname.rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "png"

    tmpdir = tempfile.mkdtemp()
    try:
        in_path = os.path.join(tmpdir, f"input.{ext}")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)

        with open(in_path, "wb") as fh:
            fh.write(data)

        cmd = [
            sys.executable, str(SCRIPT),
            "-n", model_name,
            "-i", in_path,
            "-o", out_dir,
            "--outscale", str(scale),
            "--model_path", str(model_path),
            "--tile", "128",
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(BASE_DIR),
        )

        if proc.returncode != 0:
            if proc.returncode == -9:
                err_msg = "Server ran out of memory (OOM Killed). Image is too large for the current container."
            else:
                err_msg = proc.stderr.strip()[-1000:] if proc.stderr else "(no stderr output)"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            raise HTTPException(500, detail=f"Inference failed: {err_msg}")

        out_files = sorted(pathlib.Path(out_dir).iterdir())
        if not out_files:
            raise HTTPException(500, detail="No output file produced.")

        out_path   = out_files[0]
        media_type = "image/png" if out_path.suffix.lower() == ".png" else "image/jpeg"

        with open(out_path, "rb") as fh:
            result = fh.read()

        return Response(
            content=result,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_path.name}"'},
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail="Inference timed out.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
