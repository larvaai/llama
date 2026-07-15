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

$command = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && " +
    "lib /nologo /def:`"$definition`" /out:`"$buildPath\llama.lib`" /machine:x64 && " +
    "cl /nologo /std:c++20 /utf-8 /EHsc /O2 /MT " +
    "/I`"$root\native`" /I`"$llamaSource\include`" /I`"$llamaSource\ggml\include`" /I`"$llamaSource\vendor`" " +
    "`"$root\native\model_worker_main.cpp`" `"$root\native\reasoning_phase_controller.cpp`" `"$root\native\ipc_protocol.cpp`" " +
    "/link /libpath:`"$buildPath`" llama.lib /out:`"$buildPath\model-worker-native.exe`""
cmd /c $command
if ($LASTEXITCODE -ne 0) { throw "verified-runtime native build failed" }
Copy-Item "$runtime\*.dll" $buildPath -Force
Write-Output "Built verified runtime worker: $buildPath\model-worker-native.exe"
