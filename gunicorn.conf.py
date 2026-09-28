"""Gunicorn configuration used in production (Render, Railway, Fly, Docker).

    gunicorn -c gunicorn.conf.py app:app

Why one worker?
---------------
The FAISS index lives in process memory and every mutation (index / delete /
reset) rewrites ``data/index`` on disk. Two worker processes would each keep a
private copy of the index and could overwrite each other's writes. So the app
runs **one worker with several threads** (threads share the same index and the
mutation lock in ``rag/services.py`` serialises writes).
"""

import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Keep this at 1 - see the module docstring. Scale with threads instead.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))

# Embedding a large document can take a while on a shared CPU; give it room.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5

# Load the app in the worker (not in the master) so the model is imported once
# per process and a broken import fails loudly in the logs.
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'
