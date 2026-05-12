# Put Selling Dashboard

A Streamlit dashboard for ranking cash-secured put-selling setups.

## Deploying

This is a Streamlit app, not a Flask/FastAPI app. If a host says `app.py` does not export a top-level `app`, `application`, or `handler`, that host is trying to run it as a Python web server instead of Streamlit.

Use one of these options:

- Streamlit Community Cloud: set the main file path to `app.py`.
- Render: use this repo's `render.yaml`, or set the start command to `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.
- Railway/Heroku-style hosts: use the included `Procfile`.

Vercel's Python runtime is not a good fit for this app because Streamlit needs a long-running process with WebSocket support. For Vercel, the app would need to be rebuilt as a different frontend/backend architecture.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
