"""
api/index.py
============
Vercel serverless entry point for the FastAPI backend.

Vercel's Python runtime serves every file under api/ as a serverless
function. Mangum adapts the ASGI app to the Lambda-style handler.
lifespan="off" because Vercel does not reliably run lifespan handlers —
src/api.py lazy-loads the pipeline from disk on first request instead.
"""
from mangum import Mangum

from src.api import app

handler = Mangum(app, lifespan="off")
