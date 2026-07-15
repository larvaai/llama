param(
    [Parameter(Mandatory = $true)][string]$ModelManifest,
    [string]$BuildDirectory = "build",
    [int]$SoakRequests = 500,
    [int]$Port = 8090
)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$manifestPath = [IO.Path]::GetFullPath($ModelManifest)
$buildPath = [IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
$runtimeBuildPath = "$buildPath-runtime"
$workerUrl = "http://127.0.0.1:$Port"
$service = $null
$monitor = $null
$recoveryRequest = $null
$stopFile = $null

function Write-JsonFile([string]$Path, $Value) {
    $Value | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 $Path
}

function Run-Gate([string]$Name, [scriptblock]$Command, [string]$Artifact) {
    $started = [DateTimeOffset]::UtcNow
    # Windows PowerShell 5 surfaces native stderr as non-terminating error
    # records. The top-level Stop preference would otherwise abort a healthy
    # command before its exit code can be inspected or its artifact written.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $Command 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    Write-JsonFile $Artifact @{
        gate = $Name
        status = $(if ($exitCode -eq 0) { "passed" } else { "failed" })
        exit_code = $exitCode
        started_at = $started.ToString("o")
        duration_seconds = ([DateTimeOffset]::UtcNow - $started).TotalSeconds
        output = $output
    }
    if ($exitCode -ne 0) { throw "$Name failed with exit code $exitCode" }
}

function Get-ReadySnapshot([string]$Url, [int]$TimeoutMilliseconds = 2000) {
    $response = $null
    try {
        $request = [Net.HttpWebRequest]::Create($Url)
        $request.Timeout = $TimeoutMilliseconds
        $request.ReadWriteTimeout = $TimeoutMilliseconds
        try {
            $response = $request.GetResponse()
        } catch [Net.WebException] {
            if (-not $_.Exception.Response) { return $null }
            $response = $_.Exception.Response
        }
        $reader = New-Object IO.StreamReader($response.GetResponseStream())
        try {
            $payload = $reader.ReadToEnd() | ConvertFrom-Json
        } finally {
            $reader.Dispose()
        }
        $payload | Add-Member -NotePropertyName http_status -NotePropertyValue ([int]$response.StatusCode) -Force
        return $payload
    } catch {
        return $null
    } finally {
        if ($response) { $response.Close() }
    }
}

function Assert-ReadyIdentity($Ready, [string]$Revision, $ExpectedIdentity, [string]$NativeHash) {
    foreach ($pair in @(
        @("revision", $Revision),
        @("manifest_digest", $ExpectedIdentity.manifest_digest),
        @("runtime_build", $ExpectedIdentity.runtime_build),
        @("model_digest", $ExpectedIdentity.model_digest),
        @("native_executable_sha256", $NativeHash)
    )) {
        if ($Ready.($pair[0]) -ne $pair[1]) { throw "ready identity mismatch for $($pair[0])" }
    }
}

function Get-NativeChildren([int]$ParentProcessId, [string]$ExpectedExecutable) {
    $expectedPath = [IO.Path]::GetFullPath($ExpectedExecutable)
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentProcessId")
    return @($children | Where-Object {
        $_.ExecutablePath -and
        [IO.Path]::GetFullPath($_.ExecutablePath).Equals(
            $expectedPath,
            [StringComparison]::OrdinalIgnoreCase
        )
    })
}

function Get-ExactNativeChild(
    [int]$ParentProcessId,
    [string]$ExpectedExecutable,
    [string]$ExpectedHash
) {
    $matches = @(Get-NativeChildren $ParentProcessId $ExpectedExecutable)
    if ($matches.Count -ne 1) {
        throw "expected exactly one native child using $ExpectedExecutable; found $($matches.Count)"
    }
    $child = $matches[0]
    $path = [IO.Path]::GetFullPath($child.ExecutablePath)
    $hash = "sha256:" + (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
    if ($hash -ne $ExpectedHash) { throw "native child executable hash mismatch" }
    return [PSCustomObject]@{
        pid = [int]$child.ProcessId
        parent_pid = [int]$child.ParentProcessId
        executable_path = $path
        executable_sha256 = $hash
    }
}

Push-Location $root
try {
    $dirty = git status --porcelain
    if ($LASTEXITCODE -ne 0) { throw "git status failed" }
    if ($dirty) { throw "release gate requires a clean Git revision; commit or intentionally remove all changes first" }
    $revision = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "cannot resolve release revision" }
    $evidence = Join-Path $root "release-evidence/$revision"
    if (Test-Path $evidence) { throw "release evidence already exists for revision $revision" }
    New-Item -ItemType Directory -Path $evidence | Out-Null

    python scripts/verify_runtime.py $manifestPath
    if ($LASTEXITCODE -ne 0) { throw "runtime verification failed" }
    $semanticAcceptance = "accept" + "ed"
    $forbiddenLiterals = @(
        $semanticAcceptance,
        "expected_result",
        "C:\Users",
        "248068",
        "248069",
        "simulate_worker_crash",
        "crash_for_test"
    )
    $forbiddenPattern = ($forbiddenLiterals | ForEach-Object { [regex]::Escape($_) }) -join "|"
    $forbidden = rg -n $forbiddenPattern model_worker native
    if ($LASTEXITCODE -eq 0) { throw "forbidden production literal found:`n$forbidden" }
    if ($LASTEXITCODE -gt 1) { throw "production source audit failed" }

    $expectedIdentity = python -c "import hashlib,json; from pathlib import Path; from model_worker.manifest import load_manifest; m=load_manifest(Path(r'$manifestPath')); print(json.dumps({'manifest_digest':m.digest,'runtime_build':m.raw['runtime_build'],'model_digest':m.raw['gguf_sha256']}))" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "cannot calculate manifest identity" }
    Write-JsonFile (Join-Path $evidence "manifest.json") @{
        manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
        manifest_digest = $expectedIdentity.manifest_digest
        model_digest = $expectedIdentity.model_digest
    }

    Run-Gate "unit-property" {
        python -m coverage erase
        python -m coverage run -m pytest tests/unit tests/property
        if ($LASTEXITCODE -ne 0) { return }
        # Unit/property coverage is an interim snapshot. Enforce the configured
        # threshold only after the integration suite has appended its coverage.
        python -m coverage report --fail-under=0
    } (Join-Path $evidence "unit-property.json")
    Run-Gate "fake-worker-integration" {
        python -m coverage run --append -m pytest tests/integration -m "not gpu"
        if ($LASTEXITCODE -ne 0) { return }
        python -m coverage report
    } (Join-Path $evidence "fake-worker-integration.json")
    Run-Gate "native-integration" {
        cmake -S . -B $buildPath -DBUILD_TESTING=ON
        if ($LASTEXITCODE -ne 0) { return }
        cmake --build $buildPath --config Release --target model-worker-native model-worker-native-tests
        if ($LASTEXITCODE -ne 0) { return }
        ctest --test-dir $buildPath -C Release --output-on-failure
        if ($LASTEXITCODE -ne 0) { return }
        & scripts/build_native_runtime.ps1 -ModelManifest $manifestPath -BuildDirectory "$BuildDirectory-runtime"
    } (Join-Path $evidence "native-integration.json")

    $nativeExecutable = Join-Path $runtimeBuildPath "model-worker-native.exe"
    $nativeHash = "sha256:" + (Get-FileHash -Algorithm SHA256 $nativeExecutable).Hash.ToLowerInvariant()
    Write-JsonFile (Join-Path $evidence "build.json") @{
        revision = $revision
        native_executable = $nativeExecutable
        native_executable_sha256 = $nativeHash
        runtime_build = $expectedIdentity.runtime_build
    }
    Run-Gate "real-native-fault-injection" {
        python -m pytest tests/gpu/test_native_faults.py --model-manifest $manifestPath --native-executable $nativeExecutable --require-gpu -q
    } (Join-Path $evidence "fault-injection.json")

    $env:MODEL_WORKER_REVISION = $revision
    $stdoutLog = Join-Path $evidence "service.stdout.log"
    $stderrLog = Join-Path $evidence "service.stderr.log"
    $serviceStartedAt = [DateTimeOffset]::UtcNow
    $service = Start-Process python -ArgumentList @(
        "-m", "model_worker.cli",
        "--model-manifest", $manifestPath,
        "--native-executable", $nativeExecutable,
        "--host", "127.0.0.1",
        "--port", "$Port"
    ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    $ready = $null
    $readyDeadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $readyDeadline) {
        if ($service.HasExited) { throw "exact release service exited before readiness" }
        $ready = Get-ReadySnapshot "$workerUrl/ready"
        if ($ready -and $ready.status -eq "ready" -and $ready.http_status -eq 200) { break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready -or $ready.status -ne "ready") { throw "exact release service did not become ready" }
    $initialReadyObservedAt = [DateTimeOffset]::UtcNow
    $initialStartupMs = ($initialReadyObservedAt - $serviceStartedAt).TotalMilliseconds
    Assert-ReadyIdentity $ready $revision $expectedIdentity $nativeHash
    if ($ready.process_generation -lt 1) { throw "ready process generation is invalid" }
    $initialReady = $ready
    $initialGeneration = [int]$initialReady.process_generation
    $initialNative = Get-ExactNativeChild $service.Id $nativeExecutable $nativeHash

    # Prove actual request-triggered recovery. Killing the exact native child must
    # first expose DEGRADED; no replacement may appear until the request arrives.
    $crashRequestedAt = [DateTimeOffset]::UtcNow
    Stop-Process -Id $initialNative.pid -Force
    $crashDeadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
    while ((Get-Process -Id $initialNative.pid -ErrorAction SilentlyContinue) -and
           [DateTimeOffset]::UtcNow -lt $crashDeadline) {
        Start-Sleep -Milliseconds 25
    }
    if (Get-Process -Id $initialNative.pid -ErrorAction SilentlyContinue) {
        throw "native child did not exit after crash injection"
    }
    $crashConfirmedAt = [DateTimeOffset]::UtcNow

    $degraded = $null
    $degradedDeadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
    while ([DateTimeOffset]::UtcNow -lt $degradedDeadline) {
        if ($service.HasExited) { throw "service exited after native child crash" }
        $candidate = Get-ReadySnapshot "$workerUrl/ready"
        if ($candidate -and $candidate.http_status -eq 503 -and
            $candidate.supervisor_state -eq "DEGRADED") {
            $degraded = $candidate
            break
        }
        Start-Sleep -Milliseconds 50
    }
    if (-not $degraded) { throw "service did not expose DEGRADED after native child crash" }
    $degradedObservedAt = [DateTimeOffset]::UtcNow
    if ([int]$degraded.process_generation -ne $initialGeneration) {
        throw "process generation changed before recovery request"
    }
    if (@(Get-NativeChildren $service.Id $nativeExecutable).Count -ne 0) {
        throw "native child restarted before the recovery request"
    }

    $recoveryOutput = Join-Path $evidence "restart-recovery-request.json"
    $recoveryStdout = Join-Path $evidence "restart-recovery-request.stdout.log"
    $recoveryStderr = Join-Path $evidence "restart-recovery-request.stderr.log"
    $recoveryTriggeredAt = [DateTimeOffset]::UtcNow
    $recoveryRequest = Start-Process python -ArgumentList @(
        "scripts/soak_worker.py",
        "--model-manifest", $manifestPath,
        "--requests", "1",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--output", $recoveryOutput
    ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $recoveryStdout -RedirectStandardError $recoveryStderr

    $spawnedNative = $null
    $spawnDeadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
    while ([DateTimeOffset]::UtcNow -lt $spawnDeadline) {
        if ($service.HasExited) { throw "service exited during request-triggered recovery" }
        if ($recoveryRequest.HasExited -and $recoveryRequest.ExitCode -ne 0) {
            throw "request-triggered recovery request failed with exit code $($recoveryRequest.ExitCode)"
        }
        $matches = @(Get-NativeChildren $service.Id $nativeExecutable)
        if ($matches.Count -gt 1) { throw "recovery request spawned multiple native children" }
        if ($matches.Count -eq 1) {
            $spawnedNative = Get-ExactNativeChild $service.Id $nativeExecutable $nativeHash
            break
        }
        Start-Sleep -Milliseconds 25
    }
    if (-not $spawnedNative) { throw "recovery request did not spawn the exact native child" }

    # This readiness request blocks on the supervisor lock while the newly spawned
    # child loads the model. One long request avoids filling handler slots with
    # abandoned short polls during a legitimate slow load.
    $postRestartReady = Get-ReadySnapshot "$workerUrl/ready" 180000
    $postRestartReadyObservedAt = [DateTimeOffset]::UtcNow
    if (-not $postRestartReady -or $postRestartReady.http_status -ne 200 -or
        $postRestartReady.status -ne "ready") {
        throw "request-triggered native recovery did not become ready"
    }
    Assert-ReadyIdentity $postRestartReady $revision $expectedIdentity $nativeHash
    if ([int]$postRestartReady.process_generation -ne $initialGeneration + 1) {
        throw "request-triggered recovery must increment process generation exactly once"
    }
    $restartedNative = Get-ExactNativeChild $service.Id $nativeExecutable $nativeHash
    if ($restartedNative.pid -ne $spawnedNative.pid) {
        throw "native child changed between recovery spawn and readiness"
    }
    if ($restartedNative.pid -eq $initialNative.pid) {
        throw "restarted native child unexpectedly reused the crashed PID"
    }

    $recoveryRequest | Wait-Process -Timeout 210
    if ($recoveryRequest.ExitCode -ne 0 -or -not (Test-Path $recoveryOutput)) {
        throw "recovery request did not complete successfully"
    }
    $recoveryCompletedAt = [DateTimeOffset]::UtcNow
    $recoveryResult = Get-Content -Raw $recoveryOutput | ConvertFrom-Json
    $recoveryGenerations = @($recoveryResult.process_generations)
    if (@($recoveryResult.failures).Count -ne 0 -or $recoveryGenerations.Count -ne 1 -or
        [int]$recoveryGenerations[0] -ne [int]$postRestartReady.process_generation) {
        throw "recovery request did not execute on the restarted native generation"
    }
    $recoveryRequest = $null
    $restartMs = ($postRestartReadyObservedAt - $crashConfirmedAt).TotalMilliseconds
    Write-JsonFile (Join-Path $evidence "restart-recovery.json") @{
        gate = "restart-recovery"
        status = "passed"
        initial_startup_time_ms = $initialStartupMs
        restart_time_ms = $restartMs
        degraded_observation_time_ms = ($degradedObservedAt - $crashConfirmedAt).TotalMilliseconds
        recovery_request_time_ms = ($recoveryCompletedAt - $recoveryTriggeredAt).TotalMilliseconds
        service_pid = $service.Id
        crash_requested_at = $crashRequestedAt.ToString("o")
        crash_confirmed_at = $crashConfirmedAt.ToString("o")
        degraded_observed_at = $degradedObservedAt.ToString("o")
        recovery_triggered_at = $recoveryTriggeredAt.ToString("o")
        restarted_ready_at = $postRestartReadyObservedAt.ToString("o")
        initial_identity = $initialReady
        identity_after_restart = $postRestartReady
        initial_native_process = $initialNative
        restarted_native_process = $restartedNative
        same_native_binary = (
            $initialNative.executable_sha256 -eq $restartedNative.executable_sha256 -and
            $restartedNative.executable_sha256 -eq $nativeHash
        )
    }
    $ready = $postRestartReady

    Run-Gate "gpu-model" {
        python -m pytest tests/gpu/test_model_release.py --model-manifest $manifestPath --worker-url $workerUrl --require-gpu -q
    } (Join-Path $evidence "gpu.json")

    $stopFile = Join-Path $evidence ".stop-resource-monitor"
    $resourcePath = Join-Path $evidence "resource-series.json"
    $monitor = Start-Process python -ArgumentList @(
        "scripts/monitor_resources.py", "--pid", "$($service.Id)",
        "--output", $resourcePath, "--stop-file", $stopFile
    ) -PassThru -WindowStyle Hidden
    python scripts/soak_worker.py --model-manifest $manifestPath --requests $SoakRequests --host 127.0.0.1 --port $Port --output (Join-Path $evidence "soak.json")
    if ($LASTEXITCODE -ne 0) { throw "soak gate failed" }
    New-Item -ItemType File -Path $stopFile | Out-Null
    $monitor | Wait-Process -Timeout 30
    if ($monitor.ExitCode -ne 0 -or -not (Test-Path $resourcePath)) { throw "resource monitor failed" }
    $monitor = $null

    $soak = Get-Content -Raw (Join-Path $evidence "soak.json") | ConvertFrom-Json
    $resources = Get-Content -Raw $resourcePath | ConvertFrom-Json
    if ($resources.rss_scope -ne "service_process_tree" -or
        $null -eq $resources.peak_rss_bytes_process_tree) {
        throw "resource evidence is missing aggregate service process-tree RSS"
    }
    $nativeObserved = @($resources.samples | Where-Object {
        @($_.process_tree_pids) -contains $restartedNative.pid
    }).Count -gt 0
    if (-not $nativeObserved) {
        throw "resource evidence never observed the exact restarted native child"
    }
    if ($null -eq $resources.peak_vram_mib_process_tree -and
        $null -eq $resources.peak_vram_mib_total_system_fallback) {
        throw "resource evidence has neither per-process VRAM nor labelled system fallback"
    }
    Write-JsonFile (Join-Path $evidence "summary.json") @{
        consolidated_release_gate = "passed"
        revision = $revision
        identity = $ready
        initial_identity = $initialReady
        identity_after_restart = $ready
        initial_startup_time_ms = $initialStartupMs
        restart_time_ms = $restartMs
        latency_seconds = $soak.latency_seconds
        throughput = $soak.throughput
        rss_scope = $resources.rss_scope
        peak_rss_bytes = $resources.peak_rss_bytes_process_tree
        peak_rss_bytes_process_tree = $resources.peak_rss_bytes_process_tree
        gpu_vram_measurement_scopes = $resources.gpu_vram_measurement_scopes
        peak_vram_mib_process_tree = $resources.peak_vram_mib_process_tree
        peak_vram_mib_total_system_fallback = $resources.peak_vram_mib_total_system_fallback
        peak_vram_mib_total_system = $resources.peak_vram_mib_total_system
        failure_count = $soak.failures.Count
        required_gates = @(
            "unit-property", "fake-worker-integration", "native-integration",
            "real-native-fault-injection", "restart-recovery", "gpu-model", "soak",
            "resource-series"
        )
    }
    Write-Output "Model Worker M0 release gates passed: $evidence"
} finally {
    if ($recoveryRequest -and -not $recoveryRequest.HasExited) { $recoveryRequest | Stop-Process -Force }
    if ($monitor -and -not $monitor.HasExited) {
        if ($stopFile -and -not (Test-Path $stopFile)) { New-Item -ItemType File -Path $stopFile | Out-Null }
        $monitor | Wait-Process -Timeout 5 -ErrorAction SilentlyContinue
        if (-not $monitor.HasExited) { $monitor | Stop-Process -Force }
    }
    if ($service -and -not $service.HasExited) { $service | Stop-Process -Force }
    Remove-Item Env:MODEL_WORKER_REVISION -ErrorAction SilentlyContinue
    Pop-Location
}
