# UpscaleAI — Real-ESRGAN Upscaler

Two parts: a **Railway backend** (Python/Docker) and a **standalone HTML frontend** (`upscaler.html`).

***

## Part 1 — Deploy the Backend to Railway

### Prerequisites
- Free [Railway account](https://railway.app) — no credit card required on hobby tier
- [GitHub account](https://github.com)

### Steps

**1. Create a GitHub repo**
Name it anything, e.g. `upscale-backend`. Public or private, doesn't matter.

**2. Add these three files to the repo root** (from the `backend/` folder):
```
main.py
requirements.txt
Dockerfile
```

**3. Push to GitHub.**

**4. Go to [railway.app](https://railway.app)**
→ New Project → Deploy from GitHub Repo → select your repo.

Railway detects the Dockerfile automatically and starts building.

**5. Generate a public URL**
Once the build finishes: click your service → **Settings → Networking → Generate Domain**.

Copy the URL. It looks like:
```
https://upscale-backend-production.up.railway.app
```

> **First deploy takes 8–12 minutes** — it installs PyTorch, downloads both model weights
> (~65 MB for x4plus, ~65 MB for x2plus), and compiles the container.
> Subsequent deploys are under 2 minutes.

> **Free tier note:** Railway hobby plan gives $5/mo free compute credit.
> Upscaling a few images per day fits within this easily.

***

## Part 2 — Use the Frontend

1. Open `upscaler.html` in any browser — no install, no server, just open the file
2. Paste your Railway URL into the **API** bar at the top
3. Click **Check** — the dot should turn green (Online)
4. Drop or click to upload an image (JPEG, PNG, or WebP — max 20 MB)
5. Choose **2×** or **4×** upscale
6. Click **Upscale** — preview both images side-by-side
7. Click **Download** when ready

***

## Model Details

| Scale | Model | Best for |
|-------|-------|----------|
| 4×    | RealESRGAN_x4plus | Photographs, general images |
| 2×    | RealESRGAN_x2plus | Already high-res images, subtle upscale |

Both models are downloaded during Docker build. No internet required at inference time.

***

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Dot stays red (Offline) | Railway free tier sleeps after ~10 min of inactivity. Go to your Railway dashboard, click the service to wake it, wait ~15s, then check again. |
| "Weights missing" warning | Container is still initializing. Wait 60 seconds and click Check again. |
| Request timed out | Image may be very large or the container is cold-starting. Retry once — second attempt will be faster. |
| "Unsupported format" | Browser may detect MIME type incorrectly for some files. Resave as PNG and retry. |
| CORS error in console | Confirm your Railway URL has no trailing slash and starts with `https://`. |

***

## Architecture

```
[Browser — upscaler.html]
        │  POST /upscale?scale=4
        │  multipart/form-data (image file)
        ▼
[Railway — FastAPI + Docker]
        │  subprocess → inference_realesrgan.py
        │  → RealESRGAN_x4plus.pth (fp32 CPU / fp16 CUDA)
        ▼
[Response — image/png binary]
        │  Blob URL → <img> preview
        ▼
[Download button → anchor click]
```
