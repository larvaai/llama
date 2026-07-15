param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$RuntimeManifest = "$PSScriptRoot/../runtime-manifest.json"
)
$ErrorActionPreference = "Stop"
$manifest = Get-Content -Raw $RuntimeManifest | ConvertFrom-Json
$binary = $manifest.binary_distribution
if ($binary.sha256 -eq "REQUIRED_BEFORE_FETCH") {
    throw "runtime-manifest.json must contain the audited archive SHA-256 before fetching"
}
$destinationPath = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
$archive = Join-Path $destinationPath "runtime.zip"
Invoke-WebRequest -Uri $binary.url -OutFile $archive
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
if ($actual -ne $binary.sha256.Replace("sha256:", "").ToLowerInvariant()) {
    Remove-Item -LiteralPath $archive
    throw "runtime archive SHA-256 mismatch"
}
Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $destinationPath "runtime") -Force
Remove-Item -LiteralPath $archive
