$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Model = 'C:\Users\namso\.lmstudio\models\LuffyTheFox\Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF\Qwen3.5-9B.Q4_K_M.gguf'
$Artifacts = Join-Path $Root 'artifacts'
$Tokens = Join-Path $Artifacts 'phase-c-gpu-tokens.jsonl'
$RuntimeLog = Join-Path $Artifacts 'phase-c-gpu-runtime.log'

& (Join-Path $Root 'build_phase_b.ps1') *> (Join-Path $Artifacts 'phase-c-build.log')
Remove-Item $Tokens -ErrorAction SilentlyContinue
$ErrorActionPreference = 'Continue' # llama.cpp writes normal diagnostics to stderr.
& (Join-Path $Root 'build\phase_b_sample.exe') $Model $Tokens 99 grammar *> $RuntimeLog
$nativeExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($nativeExit -ne 0) { throw "Controlled decoder failed: $nativeExit" }
$env:PYTHONIOENCODING = 'utf-8'
python (Join-Path $Root 'analyze_phase_c.py')
if ($LASTEXITCODE -ne 0) { throw "Phase C analysis failed: $LASTEXITCODE" }
