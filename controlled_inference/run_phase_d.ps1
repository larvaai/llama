$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Artifacts = Join-Path $Root 'artifacts'
$Model = 'C:\Users\namso\.lmstudio\models\LuffyTheFox\Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF\Qwen3.5-9B.Q4_K_M.gguf'
$Schema = Join-Path $Root 'phase_d_schema.json'
$Grammar = Join-Path $Artifacts 'phase-d.gbnf'
$env:PYTHONIOENCODING = 'utf-8'

python -m unittest (Join-Path $Root 'test_schema_compiler.py')
if ($LASTEXITCODE -ne 0) { throw "Schema compiler tests failed" }
python (Join-Path $Root 'compile_schema_subset.py') $Schema $Grammar *> (Join-Path $Artifacts 'phase-d-compiler.log')
if ($LASTEXITCODE -ne 0) { throw "Schema compilation failed" }
& (Join-Path $Root 'build_phase_b.ps1') *> (Join-Path $Artifacts 'phase-d-build.log')

$Cases = @('en', 'vi', 'auto_en', 'auto_vi')
foreach ($CaseId in $Cases) {
    $CaseFile = Join-Path $Root "phase_d_cases\$CaseId.json"
    $SystemPrompt = Join-Path $Artifacts "phase-d-$CaseId-system.txt"
    $UserPrompt = Join-Path $Artifacts "phase-d-$CaseId-user.txt"
    $Tokens = Join-Path $Artifacts "phase-d-$CaseId-tokens.jsonl"
    $RuntimeLog = Join-Path $Artifacts "phase-d-$CaseId-runtime.log"
    python (Join-Path $Root 'prepare_phase_d_prompt.py') $CaseFile $SystemPrompt $UserPrompt
    if ($LASTEXITCODE -ne 0) { throw "Prompt preparation failed: $CaseId" }
    Remove-Item $Tokens -ErrorAction SilentlyContinue
    $ErrorActionPreference = 'Continue'
    & (Join-Path $Root 'build\phase_b_sample.exe') $Model $Tokens 99 schema $Grammar $SystemPrompt $UserPrompt *> $RuntimeLog
    $nativeExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($nativeExit -ne 0) { throw "Controlled decoder failed for ${CaseId}: $nativeExit" }
    python (Join-Path $Root 'analyze_phase_d.py') $CaseId *> (Join-Path $Artifacts "phase-d-$CaseId-analysis.log")
    if ($LASTEXITCODE -ne 0) { throw "Phase D analysis failed: $CaseId" }
}

python (Join-Path $Root 'aggregate_phase_d_languages.py')
if ($LASTEXITCODE -ne 0) { throw "Language matrix failed" }
