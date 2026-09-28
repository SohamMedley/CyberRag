# Deploying CYBER-RAG on Render

This is the click-by-click version. Total time: about 15 minutes.
Everything below uses free-tier features only, with paid upgrades called out
where they matter.

---

## 0. Before you start

| You need | Where to get it | Cost |
| --- | --- | --- |
| A GitHub account with this repository pushed to it | [github.com](https://github.com) | free |
| A Render account (sign in with GitHub) | [render.com](https://render.com) | free |
| A Groq API key | [console.groq.com/keys](https://console.groq.com/keys) | free tier |

You do **not** need a credit card for the free plan.

> **First, get the code on GitHub.** If you are working from a branch, open a
> pull request and merge it into `main` - Render deploys whatever branch you
> point it at (the blueprint below assumes `main`).

```bash
git add -A
git commit -m "CyberRAG 2.0: Library + Ask UI, per-file delete, Render deployment"
git push origin main
```

---

## 1. Create the service

### Option A - Blueprint (recommended, renders `render.yaml`)

1. Go to <https://dashboard.render.com/blueprints> and click **New Blueprint Instance**.
2. Connect the GitHub repository containing CYBER-RAG.
3. Render reads `render.yaml` and shows a service named **cyber-rag**.
4. It asks for the value of `GROQ_API_KEY` (it is marked as a secret, so it is
   never stored in git). Paste the key from
   <https://console.groq.com/keys> - it starts with `gsk_`.
5. Click **Apply** / **Create**.

### Option B - Manual web service

1. <https://dashboard.render.com> → **New** → **Web Service** → connect the repo.
2. Fill in:

   | Field | Value |
   | --- | --- |
   | Name | `cyber-rag` |
   | Runtime | `Python 3` |
   | Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
   | Start Command | `gunicorn -c gunicorn.conf.py app:app` |
   | Health Check Path | `/healthz` |
   | Instance Type | Free |

3. Add the environment variables from the table in section 2, then click
   **Create Web Service**.

---

## 2. Environment variables

Set these under **Environment** in the Render dashboard.
Only `GROQ_API_KEY` is required - the rest have sensible defaults.

| Key | Value | Why |
| --- | --- | --- |
| `GROQ_API_KEY` | `gsk_...` | Your personal key. **Required.** |
| `PYTHON_VERSION` | `3.11.9` | Matches what the project is tested on. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Fast, high quality, free-tier friendly. |
| `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-120b,llama-3.1-8b-instant` | Used automatically when the primary model is rate limited. |
| `GROQ_REASONING_EFFORT` | `low` | Keeps answers quick and token-cheap. |
| `EMBEDDING_THREADS` | `1` | Render's CPU is shared; avoids thread oversubscription. |
| `WARMUP_EMBEDDINGS` | `true` | Downloads the embedding model during boot instead of on your first upload. |
| `LOG_LEVEL` | `INFO` | Useful while you are testing. |

Do **not** set `FLASK_DEBUG=true` in production.

---

## 3. First deploy

Watch the **Logs** tab. A healthy boot looks like:

```
CYBER-RAG.store: No persisted index found - starting with an empty library.
CYBER-RAG.services: Groq model 'openai/gpt-oss-20b' will be validated on the first question (lazy).
CYBER-RAG: CYBER-RAG 2.0.0 ready | data=/opt/render/project/src/data ...
[INFO] Listening at: http://0.0.0.0:10000
```

The first build takes **5-10 minutes** because PyTorch (CPU build, ~200 MB) is
installed. Later deploys are faster thanks to Render's build cache.

When the log says *"Your service is live"*, open the URL Render shows at the top
of the page (looks like `https://cyber-rag-xxxx.onrender.com`).

---

## 4. Smoke test the deployment

```bash
BASE=https://cyber-rag-xxxx.onrender.com

curl $BASE/healthz          # {"chunks":0,"status":"ok","version":"2.0.0"}
curl $BASE/status           # embedding model, model id, library contents
```

Then in the browser:

1. **Library** tab → drag one of the files from `Inputs & Ques Sample/` into the upload box.
2. Click **Index documents**. The first index also downloads the 90 MB embedding
   model, so give it ~1-2 minutes on the free instance.
3. Switch to **Ask**, type a question about the document you indexed, and send it.
4. Press the trash icon next to a document to verify delete works.

---

## 5. Free plan: what to expect

| Behaviour | Free plan | Starter plan ($7/mo) |
| --- | --- | --- |
| Sleeps after 15 min idle | yes - first visit takes ~30-60 s to wake | no |
| Disk | **ephemeral** - index and uploads are erased on every deploy/restart | persistent disk supported |
| RAM / CPU | 512 MB / shared | 512 MB+ / more headroom |
| Custom domain, HTTPS | yes | yes |

On the free plan the vector index is rebuilt from scratch after every deploy or
spin-down, so keep your source documents handy and re-index when needed.

### Making data persist (Starter plan)

1. Dashboard → your service → **Disks** → **Add Disk**:
   name `cyberrag-data`, mount path `/var/data`, size `1 GB`.
2. Add these environment variables:

   ```
   DATA_DIR=/var/data
   UPLOAD_DIR=/var/data/uploads
   HF_HOME=/var/data/hf-cache
   ```

3. Save - Render redeploys automatically. The index, your uploads and the
   downloaded embedding model now survive restarts. `render.yaml` contains the
   same block commented out; uncomment it (and switch `plan: starter`) if you
   prefer Blueprint-managed settings.

---

## 6. Redeploying after code changes

With `autoDeploy: true` (the default), every push to the connected branch
triggers a new deploy:

```bash
git add -A && git commit -m "your change" && git push origin main
```

You can also hit **Manual Deploy → Deploy latest commit** in the dashboard.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Build fails on `torch==2.4.0+cpu` | The PyTorch CPU index was unreachable | Retry the deploy; check Render status page. |
| `503 The embedding model is not available yet` | The model could not be downloaded | Confirm outbound internet works, or set `WARMUP_EMBEDDINGS=true` and check logs; with a disk, pre-populate `HF_HOME`. |
| `502 ... Groq request failed` | Upstream/API error or all fallback models exhausted | Check your Groq quota at console.groq.com; the response body contains the reason. |
| `429` / "rate limit" message in the answer | Free-tier tokens-per-minute ceiling | Wait a minute, or set `GROQ_MODEL` to a lighter model such as `openai/gpt-oss-20b`. |
| First question is slow | Cold wake-up + model load + Groq call | Expected on free; later questions are fast. |
| `CYBER-RAG expects groq>=1.0` | An old `groq` package was installed | Rebuild without cache: **Manual Deploy → Clear build cache & deploy**. |
| Service is up but the page is blank | Stale JS/CSS in the browser cache | Hard-refresh (Ctrl/Cmd+Shift+R). |

---

## 8. Rolling back

Dashboard → **Deploys** → pick an earlier successful deploy → **Redeploy**.
Every deploy is kept, so rollback takes seconds.
