param(
    [Parameter(Mandatory = $true)][string]$ModelManifest,
    [string]$BuildDirectory = "build-runtime"
)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$manifestPath = [IO.Path]::GetFullPath($ModelManifest)
$buildPath = [IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
python (Join-Path $PSScriptRoot "verify_runtime.py") $manifestPath
if ($LASTEXITCODE -ne 0) { throw "runtime verification failed" }
$manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
$runtime = [IO.Path]::GetFullPath($manifest.runtime.directory)
$llamaSource = Join-Path $root "controlled_inference/vendor/llama.cpp"
$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) { throw "Visual Studio 2022 Build Tools are required" }
New-Item -ItemType Directory -Force $buildPath | Out-Null

# The DLL hash was verified above before its exports are trusted.
$exports = cmd /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && dumpbin /nologo /exports `"$runtime\llama.dll`""
$names = $exports | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*$') { $Matches[1] }
}
if (-not $names) { throw "verified llama.dll exported no symbols" }
$definition = Join-Path $buildPath "llama.def"
@("LIBRARY llama.dll", "EXPORTS") + ($names | ForEach-Object { "    $_" }) | Set-Content -Encoding ascii $definition

$commonExports = cmd /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && dumpbin /nologo /exports `"$runtime\llama-common.dll`""
$commonNames = $commonExports | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*$') { $Matches[1] }
}
if (-not $commonNames) { throw "verified llama-common.dll exported no symbols" }
$commonDefinition = Join-Path $buildPath "llama-common.def"
@("LIBRARY llama-common.dll", "EXPORTS") + ($commonNames | ForEach-Object { "    $_" }) | Set-Content -Encoding ascii $commonDefinition

$workerCommand = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && " +
    "lib /nologo /def:`"$definition`" /out:`"$buildPath\llama.lib`" /machine:x64 && " +
    "lib /nologo /def:`"$commonDefinition`" /out:`"$buildPath\llama-common.lib`" /machine:x64 && " +
    "cl /nologo /std:c++20 /utf-8 /EHsc /O2 /MT " +
    "/I`"$root\native`" /I`"$llamaSource\include`" /I`"$llamaSource\ggml\include`" /I`"$llamaSource\vendor`" " +
    "`"$root\native\model_worker_main.cpp`" `"$root\native\generation_safety.cpp`" `"$root\native\pending_cancel_registry.cpp`" `"$root\native\reasoning_phase_controller.cpp`" `"$root\native\sequence_engine.cpp`" `"$root\native\ipc_protocol.cpp`" " +
    "/link /libpath:`"$buildPath`" llama.lib /out:`"$buildPath\model-worker-native.exe`""
cmd /c $workerCommand
if ($LASTEXITCODE -ne 0) { throw "verified-runtime native build failed" }
$runtimeCommand = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && " +
    "cl /nologo /std:c++20 /utf-8 /EHsc /O2 /MT " +
    "/I`"$root\native`" /I`"$llamaSource\include`" /I`"$llamaSource\ggml\include`" /I`"$llamaSource\vendor`" /I`"$llamaSource\common`" " +
    "`"$root\native\inference_runtime_main.cpp`" `"$root\native\generation_safety.cpp`" `"$root\native\reasoning_phase_controller.cpp`" `"$root\native\sequence_engine.cpp`" " +
    "/link /libpath:`"$buildPath`" llama.lib llama-common.lib /out:`"$buildPath\inference-runtime-native.exe`""
cmd /c $runtimeCommand
if ($LASTEXITCODE -ne 0) { throw "verified-runtime inference scheduler build failed" }
Copy-Item "$runtime\*.dll" $buildPath -Force
Write-Output "Built verified runtime worker: $buildPath\model-worker-native.exe"
Write-Output "Built verified inference runtime: $buildPath\inference-runtime-native.exe"
