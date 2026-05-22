# Files you need

Put these four backend files in the root of your GitHub repo:

- Dockerfile
- main.py
- requirements.txt
- railway.json

Frontend file:
- upscaler.html

Railway uses `Dockerfile` at the repo root by default, and Docker deployments use the Dockerfile's CMD/ENTRYPOINT unless overridden. Railway docs also note that environment variables in Docker start commands require wrapping the command in a shell for expansion. [page:1][page:0]
