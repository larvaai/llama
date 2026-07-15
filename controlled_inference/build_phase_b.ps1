$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'vendor\official-b10012\runtime'
$Source = Join-Path $Root 'vendor\llama.cpp'
$Build = Join-Path $Root 'build'
$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
New-Item -ItemType Directory -Force $Build | Out-Null

$def = Join-Path $Build 'llama.def'
$exports = cmd /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && dumpbin /nologo /exports `"$Runtime\llama.dll`""
$names = $exports | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*$') { $Matches[1] }
}
@('LIBRARY llama.dll', 'EXPORTS') + ($names | ForEach-Object { "    $_" }) | Set-Content -Encoding ascii $def

$command = "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && " +
    "lib /nologo /def:`"$def`" /out:`"$Build\llama.lib`" /machine:x64 && " +
    "cl /nologo /std:c++17 /utf-8 /EHsc /O2 /I`"$Source\include`" /I`"$Source\ggml\include`" `"$Root\phase_b_sample.cpp`" " +
    "/link /libpath:`"$Build`" llama.lib /out:`"$Build\phase_b_sample.exe`""
cmd /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }

$persistentCommand = "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && " +
    "cl /nologo /std:c++17 /utf-8 /EHsc /O2 /I`"$Source\include`" /I`"$Source\ggml\include`" `"$Root\persistent_worker.cpp`" " +
    "/link /libpath:`"$Build`" llama.lib /out:`"$Build\persistent_worker.exe`""
cmd /c $persistentCommand
if ($LASTEXITCODE -ne 0) { throw "Persistent worker build failed: $LASTEXITCODE" }

Copy-Item "$Runtime\*.dll" $Build -Force
Write-Host "Built $Build\phase_b_sample.exe and $Build\persistent_worker.exe"
