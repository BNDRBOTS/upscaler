from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import subprocess, tempfile, os, shutil, pathlib, sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Disposition"],
)

BASE_DIR = pathlib.Path(__file__).parent
SCRIPT = BASE_DIR / "inference_realesrgan.py"
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_X4 = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
MODEL_X2 = WEIGHTS_DIR / "RealESRGAN_x2plus.pth"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


USE_HALF = _cuda_available()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_x4_ready": MODEL_X4.exists(),
        "model_x2_ready": MODEL_X2.exists(),
        "cuda": USE_HALF,
    }


@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Query(default=4, ge=2, le=4)):
    if scale not in (2, 4):
        raise HTTPException(400, detail="scale must be 2 or 4")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="File exceeds 20 MB limit.")

    model_name = "RealESRGAN_x4plus" if scale == 4 else "RealESRGAN_x2plus"
    model_path = MODEL_X4 if scale == 4 else MODEL_X2
    if not model_path.exists():
        raise HTTPException(503, detail=f"Model weights not found: {model_path.name}. Container may still be initializing.")

    raw_ext = (file.filename or "input.png").rsplit('.', 1)
    ext = raw_ext[-1].lower() if len(raw_ext) == 2 and raw_ext[-1].lower() in ("jpg", "jpeg", "png", "webp") else "png"

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
        ]
        if USE_HALF:
            cmd.append("--half")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=150,
            cwd=str(BASE_DIR),
        )

        if proc.returncode != 0:
            stderr_tail = proc.stderr[-600:] if proc.stderr else "(no stderr)"
            raise HTTPException(500, detail=f"Inference error: {stderr_tail}")

        out_files = sorted(pathlib.Path(out_dir).iterdir())
        if not out_files:
            raise HTTPException(500, detail="No output file was produced.")

        out_path = out_files[0]
        media_type = "image/png" if out_path.suffix.lower() == ".png" else "image/jpeg"
        out_name = out_path.name

        with open(out_path, "rb") as fh:
            out_data = fh.read()

        return Response(
            content=out_data,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail="Inference timed out. Try a smaller image.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
