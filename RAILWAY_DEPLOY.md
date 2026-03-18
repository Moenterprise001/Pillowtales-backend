# PillowTales Production Backend Deployment Guide

## Quick Deploy to Railway (5-10 minutes)

### Step 1: Create Railway Account & Project

1. Go to **https://railway.app**
2. Sign up with GitHub (recommended) or email
3. Click **"New Project"**
4. Select **"Empty Project"**

### Step 2: Add a Service

1. In your new project, click **"+ New"**
2. Select **"GitHub Repo"** if you've pushed the code to GitHub
   - OR select **"Empty Service"** → then use **"Deploy from Local"**

### Step 3: Upload Backend Code

**Option A: GitHub (Recommended)**
1. Push the backend folder to a GitHub repo
2. Connect the repo in Railway
3. Set root directory to `/backend` (if backend is in subfolder)

**Option B: Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Navigate to backend folder
cd backend

# Link to your project
railway link

# Deploy
railway up
```

### Step 4: Add Environment Variables

In Railway Dashboard → Your Service → **Variables**, add:

```
SUPABASE_URL=https://mgclekcuskkgfnffvpdj.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your_full_service_role_key>
EMERGENT_LLM_KEY=<your_emergent_llm_key>
ELEVENLABS_API_KEY=<your_elevenlabs_api_key>
CORS_ORIGINS=*
```

> ⚠️ **Important:** Use the FULL keys, not the masked versions shown above.

### Step 5: Configure Build Settings

Railway should auto-detect Python. If needed, set:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`

### Step 6: Generate Domain

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. You'll get a URL like: `pillowtales-backend-production.up.railway.app`

**Or add custom domain:**
- Add: `api.pillowtales.co`
- Configure DNS CNAME in Cloudflare

### Step 7: Verify Deployment

Test the health endpoint:
```bash
curl https://YOUR-RAILWAY-URL.up.railway.app/api/health
```

Should return:
```json
{"status": "healthy", "service": "PillowTales API"}
```

---

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Supabase project URL | ✅ |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (full access) | ✅ |
| `EMERGENT_LLM_KEY` | OpenAI/Anthropic API key for story generation | ✅ |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for narration | ✅ |
| `CORS_ORIGINS` | Allowed origins (use `*` for mobile) | ✅ |
| `PORT` | Server port (Railway sets this automatically) | Auto |

---

## After Deployment: Update Frontend

Once you have the Railway URL, share it with me and I'll update the frontend code to use:
```
https://api.pillowtales.co/api
```
or
```
https://pillowtales-backend.up.railway.app/api
```

---

## Estimated Costs

| Usage | Monthly Cost |
|-------|--------------|
| Light (testing) | $0-5 |
| Moderate (100-500 users) | $5-15 |
| Heavy (1000+ users) | $15-50 |

Railway offers $5 free credit for new accounts.

---

## Troubleshooting

### "Build failed"
- Check requirements.txt is valid
- Ensure Python version compatibility (3.11+)
- Check Railway build logs for specific errors

### "502 Bad Gateway"
- App is still starting - wait 30 seconds
- Check start command is correct
- Verify environment variables are set

### "Connection refused"
- Ensure PORT environment variable is used (not hardcoded)
- Check the app binds to `0.0.0.0`, not `localhost`

---

## Files Included

- `server.py` - Main FastAPI application
- `requirements.txt` - Python dependencies  
- `Procfile` - Deployment start command
- `railway.json` - Railway configuration
- `nixpacks.toml` - Build configuration (includes ffmpeg)
