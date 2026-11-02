"""Loopback-only FastAPI transport for the existing Whatiff workflow."""

import subprocess

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .baseten import load_baseten_key, BasetenError
from .interface import InterfaceService
from .meta import MetaError, load_key

PROJECT = Path(__file__).resolve().parents[1]


def create_app(project=PROJECT, data=None):
    service = InterfaceService(project, data)

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.executor.shutdown(wait=True)

    app = FastAPI(title="Whatiff", lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin not in ("http://127.0.0.1:8790", "http://localhost:8790", "http://127.0.0.1:5173", "http://localhost:5173"):
                return JSONResponse({"detail": "Only local application requests are accepted."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({"detail": "Saved run or image not found."}, status_code=404)

    @app.exception_handler(MetaError)
    async def meta_error(request, error):
        return JSONResponse({"detail": error.code}, status_code=400)

    @app.get("/api/status")
    def status():
        try:
            load_key(Path(project) / ".env")
            configured = True
        except MetaError:
            configured = False
        try:
            load_baseten_key(Path(project) / ".env")
            baseten_configured = True
        except BasetenError:
            baseten_configured = False
        return {"meta_configured": configured, "baseten_configured": baseten_configured, "busy": service.busy}

    @app.post("/api/replay")
    def replay():
        return service.replay()

    @app.post("/api/runs", status_code=202)
    async def start(task: str = Form(...), backend: str = Form("controlled"), image: Optional[UploadFile] = File(None), policy_provider: str = Form("baseten")):
        content = await image.read(20 * 1024 * 1024 + 1) if image else None
        try:
            return service.start(task, backend, content, policy_provider)
        except MetaError:
            raise
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None
        except (OSError, SyntaxError):
            raise HTTPException(400, "Upload a valid PNG, JPEG, or WebP image.") from None

    @app.get("/api/runs/{run_id}")
    def get(run_id: str):
        return service.get(run_id)

    @app.post("/api/runs/{run_id}/retry", status_code=202)
    def retry(run_id: str):
        try:
            return service.retry(run_id)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/runs/{run_id}/images/{name}")
    def image(run_id: str, name: str):
        path = service.asset(run_id, name)
        if not path.is_file():
            raise HTTPException(404, "Image not found.")
        # Uploaded JPEG/WebP originals retain their bytes despite their local name.
        from .images import decode
        return FileResponse(path, media_type=decode(path.read_bytes())[1])

    @app.post("/api/runs/{run_id}/cases/{case_id}/review")
    async def review(run_id: str, case_id: str, request: Request):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Review must be an object.")
        return service.review(run_id, case_id, data)

    @app.post("/api/runs/{run_id}/reveal")
    def reveal(run_id: str):
        service.get(run_id)
        folder = service.folder(run_id).resolve()
        try:
            subprocess.run(["open", str(folder)], check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            raise HTTPException(500, "Could not open Finder. Use the displayed path.") from None
        return {"path": str(folder)}

    @app.get("/api/runs/{run_id}/report")
    def export(run_id: str):
        state = service.get(run_id)
        if state["status"] != "completed":
            raise HTTPException(409, "The evaluation report is not ready.")
        return JSONResponse(service.compact_export(run_id), headers={"Content-Disposition": f'attachment; filename="whatiff-{run_id}.json"'})

    dist = Path(project) / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8790)


if __name__ == "__main__":
    main()
