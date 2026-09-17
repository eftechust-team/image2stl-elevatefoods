# Run the Flask app via Waitress for production-like concurrency on Windows
# Usage: Open PowerShell in project root and run: .\run_waitress.ps1

$python = Join-Path -Path $PSScriptRoot -ChildPath '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host "Virtualenv python not found at $python. Activate your venv or adjust the script." -ForegroundColor Yellow
}

Write-Host "Starting Waitress server on port 8080 with 20 threads..."
& $python -m waitress --listen=*:8080 --threads=20 app:app
