"""
Launcher da API local.

Uso:
    python run_api.py
    python run_api.py --port 8080
"""

from __future__ import annotations

import argparse

import uvicorn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF RAG API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("pdf_rag.views.api:app", host=args.host, port=args.port, reload=True)
