# Tarmac — Flight Delay Predictor (website)

A static website that predicts the probability a U.S. flight departs more than
15 minutes late, **2 hours before departure**. The XGBoost model runs entirely
in the browser (exported to `assets/model.json`) — no server, no cost.

## Run locally
Any static server works, e.g.:
```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```
(Opening index.html via file:// won't work — the browser blocks fetch of the
JSON assets. Use a local server, or just deploy.)

## Deploy free on GitHub Pages (public URL, any device)
1. Commit and push the site:
   ```bash
   git add docs
   git commit -m "Add Tarmac website"
   git push
   ```
2. On github.com → your repo → **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Branch: **main**, folder: **/docs** → **Save**.
5. Wait ~1 minute. Your site is live at:
   `https://hanisohaan10.github.io/flight-delay-prediction/`

Share that link — it works on any phone or laptop, no install.

## Optional: enable "Sign in with Google"
The site works fully without it. To turn it on:
1. Google Cloud Console → APIs & Services → Credentials → create an
   **OAuth client ID** (type: Web).
2. Add your Pages URL as an **Authorised JavaScript origin**.
3. Paste the client ID into `CONFIG.googleClientId` at the top of `app.js`.

## Files
- `index.html` — page structure
- `styles.css` — styling (cream + olive theme)
- `app.js` — model inference, UI, charts, optional auth
- `assets/` — model trees + reference data + data-insight aggregates
