$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $Root 'test_phase_g_persistent.py')
