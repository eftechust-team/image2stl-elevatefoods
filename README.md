# AI Image to STL Generator

Generate multi-layer STL files from AI-generated black-and-white images with lasso/wand selection and adjustable edge dilation.

## Features

- AI image generation via Doubao API.
- Per-browser image quota capped at 4 generated images.
- Canvas editing (pen, bucket, undo/redo).
- Selection tools: Magic Wand and Lasso with add/subtract.
- Expand Selection (px) to close gaps without cumulative growth.
- Per-layer height and stacking options; STL export as a ZIP.
- Automatic upload of generated STL/ZIP files to Supabase Storage (when configured).

## Local Run

1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Set the environment variable for the Doubao API key (optional locally):
	```bash
	$env:DOUBAO_API_KEY="<your-key>"   # PowerShell
	```
3. (Optional) Configure Supabase auto-upload:
	```bash
	$env:SUPABASE_URL="https://<project-ref>.supabase.co"
	$env:SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
	$env:SUPABASE_STORAGE_BUCKET="gen-stl-zip"
	```
	Paste the raw service-role key value from Supabase here. Do not wrap it in extra quotes or add spaces/newlines.
4. Start the app:
	```bash
	python app.py
	```
5. Open http://localhost:8080

## Deploy on Render

This repo includes `render.yaml` for one-click deploy.

- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`
- Environment variables:
  - `DOUBAO_API_KEY`: your Doubao API key (set in Render Dashboard → Environment)
	- `FLASK_SECRET_KEY`: stable secret used for per-session image quotas
	- `SUPABASE_URL`: your Supabase project URL
	- `SUPABASE_SERVICE_ROLE_KEY`: service-role key used to upload files to Storage
	- `SUPABASE_STORAGE_BUCKET`: bucket name for generated files (default: `gen-stl-zip`)
	- `SUPABASE_UPLOAD_TIMEOUT`: upload timeout in seconds (default: `30`)
  - `PYTHON_VERSION`: 3.11 (provided in render.yaml)

Optional tuning:

- `MAX_IMAGES_PER_SESSION`: default `4`
- `MAX_CONCURRENT_REQUESTS`: default `2`
- `DOUBAO_REQUEST_TIMEOUT`: default `90`

Steps:
- Create a new Web Service from this repo.
- Render will detect `render.yaml` and configure the service.
- Set `DOUBAO_API_KEY` in the service’s Environment tab (as a Secret).
- Set Supabase variables (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`) if you want automatic cloud uploads.
- Paste the raw `SUPABASE_SERVICE_ROLE_KEY` value from Supabase. If the key was copied with quotes or line breaks, upload auth can fail with `InvalidCompactJWS`.
- Deploy. Auto-deploy is enabled.

## Notes

- The Doubao API key must be provided via `DOUBAO_API_KEY`. If missing, image generation returns a clear error.
- Automatic Supabase upload is enabled only when all Supabase environment variables are set.
- Logo at `/static/logo.jpeg` is optional and can be added later.