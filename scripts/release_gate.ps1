param(
    [Parameter(Mandatory = $true)][string]$ModelManifest,
    [string]$BuildDirectory = "build",
    [int]$SoakRequests = 500,
    [string]$WorkerUrl = "http://127.0.0.1:8090",
    [switch]$SkipRunningSoak
)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$manifest = [IO.Path]::GetFullPath($ModelManifest)
Push-Location $root
try {
    python scripts/verify_runtime.py $manifest
    $semanticAcceptance = "accept" + "ed"
    $forbiddenPattern = "$semanticAcceptance|expected_result|C:\\Users|248068|248069|simulate_worker_crash|crash_for_test"
    $forbidden = rg -n $forbiddenPattern model_worker native
    if ($LASTEXITCODE -eq 0) { throw "forbidden production literal found:`n$forbidden" }
    if ($LASTEXITCODE -gt 1) { throw "production source audit failed" }
    python -m pytest tests/unit tests/property
    python -m pytest tests/integration -m "not gpu"
    cmake -S . -B $BuildDirectory -DBUILD_TESTING=ON
    cmake --build $BuildDirectory --config Release --target model-worker-native model-worker-native-tests
    ctest --test-dir $BuildDirectory -C Release --output-on-failure
    & scripts/build_native_runtime.ps1 -ModelManifest $manifest -BuildDirectory "$BuildDirectory-runtime"
    python -m pytest tests/gpu --model-manifest $manifest --worker-url $WorkerUrl --require-gpu
    if (-not $SkipRunningSoak) { python scripts/soak_worker.py --model-manifest $manifest --requests $SoakRequests }
    Write-Output "Model Worker v1 release gates passed"
} finally { Pop-Location }
