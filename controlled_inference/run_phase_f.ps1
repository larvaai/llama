$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Root 'build_phase_b.ps1') *> (Join-Path $Root 'artifacts\phase-f-build.log')
$env:PYTHONIOENCODING = 'utf-8'
python (Join-Path $Root 'controlled_service.py') --host 127.0.0.1 --port 8090
