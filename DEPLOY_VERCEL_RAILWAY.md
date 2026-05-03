# Vercel + Railway (Laptop-Independent)

This setup keeps the app running even when your laptop is switched off.

## 1) Deploy backend on Railway

- Create new Railway project from the `backend` folder.
- Build/Run is already prepared:
  - `backend/Procfile`
  - `backend/Dockerfile`
- Set environment variable on Railway:
  - `DATABASE_URL` (optional, defaults to SQLite in container)
- Deploy and copy your public backend URL, for example:
  - `https://your-backend.up.railway.app`

## 2) Deploy frontend on Vercel

- Deploy from the `frontend` folder as a static project.
- After deployment, open the URL once with the backend URL in query:
  - `https://your-frontend.vercel.app/?api=https://your-backend.up.railway.app`
- This stores backend URL in browser local storage for future visits.

## 3) Share QR to everyone

- Generate QR from Vercel URL:
  - `https://your-frontend.vercel.app`
- Anyone can scan and use their own mobile camera for detection.

## Notes for stable vehicle detection

- In app, keep mode as:
  - `Smooth` for crowd/demo throughput
  - `Accurate` when showing number plate reads
- First request may be slower due to model warmup.
- Mobile browsers must allow camera permission.

## Local laptop mode still safe

- Local run stays unchanged:
  - frontend: `http://localhost:3000`
  - backend: `http://localhost:5000`
- Cloud config does not block local workflow.
