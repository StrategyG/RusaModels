[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-RusaDotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        $match = [regex]::Match($trimmed, '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
        if (-not $match.Success) { continue }

        $name = $match.Groups[1].Value
        $value = $match.Groups[2].Value.Trim()
        if ($value.Length -ge 2) {
            $first = $value.Substring(0, 1)
            $last = $value.Substring($value.Length - 1, 1)
            if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        $existing = [Environment]::GetEnvironmentVariable($name, "Process")
        if ([string]::IsNullOrWhiteSpace($existing)) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

function Wait-HttpReady {
    param(
        [string]$Uri,
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds,
        [string]$ServiceName
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "$ServiceName exited with code $($Process.ExitCode). Inspect .local/logs."
        }
        try {
            $null = Invoke-RestMethod -Uri $Uri -Method Get -TimeoutSec 3
            return
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    throw "$ServiceName did not become ready within $TimeoutSeconds seconds. Inspect .local/logs."
}

function Assert-PortAvailable {
    param([int]$Port)
    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        throw "TCP port $Port is already in use. Stop the existing service before starting RUSA-CPU-16."
    }
}

if (-not ("RusaCpuFeatureProbe" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class RusaCpuFeatureProbe
{
    [DllImport("kernel32.dll", ExactSpelling = true)]
    public static extern bool IsProcessorFeaturePresent(uint processorFeature);
}
'@
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
Push-Location $RepoRoot
$llamaProcess = $null
$apiProcess = $null
$stateDir = Join-Path $RepoRoot ".local\rusa-cpu-16"

try {
    Import-RusaDotEnv -Path (Join-Path $RepoRoot ".env")

    if ($env:RUSAMODELS_ENV -ne "edge") { throw "RUSA-CPU-16 requires RUSAMODELS_ENV=edge." }
    if ($env:DATABASE_URL -notmatch '^sqlite:///') { throw "RUSA-CPU-16 requires a local SQLite DATABASE_URL." }
    if ($env:STORAGE_BACKEND -ne "local") { throw "RUSA-CPU-16 requires STORAGE_BACKEND=local." }
    if ($env:RUSAMODELS_INFERENCE_BASE_URL -ne "http://127.0.0.1:8081") {
        throw "Inference must target http://127.0.0.1:8081."
    }
    if ($env:RUSAMODELS_INFERENCE_MODEL_ID -ne "rusa-cpu-4b") {
        throw "The model ID must be rusa-cpu-4b."
    }
    if (-not [string]::IsNullOrWhiteSpace($env:RUSAMODELS_EMBEDDING_BASE_URL) -or
        -not [string]::IsNullOrWhiteSpace($env:RUSAMODELS_EMBEDDING_MODEL_ID)) {
        throw "RUSA-CPU-16 v1.0 does not start a separate embedding runtime."
    }
    if ($env:RUSAMODELS_SERVICE_TOKEN_SHA256 -notmatch '^[0-9a-f]{64}$') {
        throw "Set RUSAMODELS_SERVICE_TOKEN_SHA256 to a lowercase SHA-256 digest."
    }
    if ($env:RUSAMODELS_API_PRINCIPALS -notmatch '^[{]') {
        throw "Set RUSAMODELS_API_PRINCIPALS to JSON containing an inference principal."
    }
    $principals = ConvertFrom-Json -InputObject $env:RUSAMODELS_API_PRINCIPALS
    $inferencePrincipals = @($principals.PSObject.Properties | Where-Object { $_.Value.role -eq "inference" })
    if ($inferencePrincipals.Count -lt 1) {
        throw "RUSAMODELS_API_PRINCIPALS must contain a role=inference principal."
    }

    if (-not $env:RUSAMODELS_GGUF_MODEL_PATH -or
        -not (Test-Path -LiteralPath $env:RUSAMODELS_GGUF_MODEL_PATH -PathType Leaf)) {
        throw "RUSAMODELS_GGUF_MODEL_PATH must point to a local GGUF file."
    }
    if ($env:RUSAMODELS_GGUF_MODEL_SHA256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "RUSAMODELS_GGUF_MODEL_SHA256 must be a 64-character SHA-256 digest."
    }
    $modelPath = (Resolve-Path -LiteralPath $env:RUSAMODELS_GGUF_MODEL_PATH).Path

    $pythonCommand = Get-Command python.exe -ErrorAction Stop
    $pythonExe = $pythonCommand.Source
    if (-not $env:LLAMA_SERVER_EXE) {
        $llamaCommand = Get-Command llama-server.exe -ErrorAction Stop
        $llamaExe = $llamaCommand.Source
    }
    else {
        if (-not (Test-Path -LiteralPath $env:LLAMA_SERVER_EXE -PathType Leaf)) {
            throw "LLAMA_SERVER_EXE does not exist: $env:LLAMA_SERVER_EXE"
        }
        $llamaExe = (Resolve-Path -LiteralPath $env:LLAMA_SERVER_EXE).Path
    }

    $processors = @(Get-CimInstance -ClassName Win32_Processor)
    $physicalCores = [int](($processors | Measure-Object -Property NumberOfCores -Sum).Sum)
    $logicalProcessors = [int](($processors | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)
    $computer = Get-CimInstance -ClassName Win32_ComputerSystem
    $memoryGiB = [math]::Round(($computer.TotalPhysicalMemory / 1GB), 2)
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $freeMemoryGiB = [math]::Round((($os.FreePhysicalMemory * 1KB) / 1GB), 2)

    if ($os.Caption -notmatch "Windows 11") {
        throw "RUSA-CPU-16 requires Windows 11 64-bit; detected $($os.Caption)."
    }
    if ($memoryGiB -lt 15) { throw "RUSA-CPU-16 expects a nominal 16-GB host; detected $memoryGiB GiB." }
    if ($freeMemoryGiB -lt 2) { throw "Only $freeMemoryGiB GiB RAM is free; minimum is 2 GiB." }
    $cpuNames = @($processors | ForEach-Object { $_.Name })
    if (($cpuNames | Where-Object { $_ -notmatch 'Intel.*Core.*i5' }).Count -gt 0) {
        throw "RUSA-CPU-16 requires an Intel Core i5 processor; detected: $($cpuNames -join '; ')."
    }
    if ($physicalCores -lt 1) { throw "Could not detect physical CPU cores." }
    $threads = [math]::Min(4, $physicalCores)
$featureMap = [ordered]@{
    SSE = 6
    SSE2 = 10
    SSE3 = 13
    SSSE3 = 36
    SSE4_1 = 37
    SSE4_2 = 38
    AVX = 39
    AVX2 = 40
    AVX512F = 41
    BMI2 = 60
}
$availableCpuFeatures = @()
foreach ($featureName in $featureMap.Keys) {
    if ([RusaCpuFeatureProbe]::IsProcessorFeaturePresent([uint32]$featureMap[$featureName])) {
        $availableCpuFeatures += $featureName
    }
}
if ($availableCpuFeatures.Count -eq 0) {
    $availableCpuFeatures = @("not_reported_by_windows")
}


    $logDir = Join-Path $RepoRoot ".local\logs"
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$hardwareProfile = [ordered]@{
    captured_utc = [datetime]::UtcNow.ToString("o")
    operating_system = $os.Caption
    os_version = $os.Version
    os_build = $os.BuildNumber
    cpu_names = $cpuNames
    physical_cores = $physicalCores
    logical_processors = $logicalProcessors
    instruction_features_reported_by_windows = $availableCpuFeatures
    total_memory_gib = $memoryGiB
    free_memory_at_start_gib = $freeMemoryGiB
}
$hardwareProfile | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stateDir "hardware-profile.json")

    foreach ($service in @("llama-server", "model-server")) {
        if (Test-Path -LiteralPath (Join-Path $stateDir "$service.json")) {
            throw "A $service process record exists. Run scripts/stop_rusa_cpu_16.ps1 first."
        }
    }

    & $pythonExe scripts/verify_gguf_model.py --model $modelPath --sha256 $env:RUSAMODELS_GGUF_MODEL_SHA256
    if ($LASTEXITCODE -ne 0) { throw "GGUF checksum verification failed." }

    Assert-PortAvailable -Port 8081
    Assert-PortAvailable -Port 8090

    & $pythonExe -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) { throw "Alembic migration failed." }

    $llamaArgs = @(
        "--model", ('"' + $modelPath + '"'),
        "--alias", "rusa-cpu-4b",
        "--host", "127.0.0.1",
        "--port", "8081",
        "--ctx-size", "8192",
        "--threads", "$threads",
        "--threads-batch", "4",
        "--threads-http", "2",
        "--parallel", "1",
        "--batch-size", "512",
        "--ubatch-size", "128",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--n-gpu-layers", "0",
        "--flash-attn", "auto",
        "--jinja",
        "--reasoning-budget", "1536",
        "--no-webui",
        "--metrics"
    )
    $llamaStart = @{
        FilePath = $llamaExe
        ArgumentList = ($llamaArgs -join " ")
        WorkingDirectory = $RepoRoot
        RedirectStandardOutput = (Join-Path $logDir "llama-server.stdout.log")
        RedirectStandardError = (Join-Path $logDir "llama-server.stderr.log")
        PassThru = $true
    }
    $llamaProcess = Start-Process @llamaStart
    @{
        pid = $llamaProcess.Id
        executable = $llamaExe
        started_utc = [datetime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateDir "llama-server.json")

    Wait-HttpReady -Uri "http://127.0.0.1:8081/health" -Process $llamaProcess -TimeoutSeconds 300 -ServiceName "llama.cpp"

    $apiStart = @{
        FilePath = $pythonExe
        ArgumentList = @("-m", "uvicorn", "services.model_server.app.main:app", "--host", "127.0.0.1", "--port", "8090")
        WorkingDirectory = $RepoRoot
        RedirectStandardOutput = (Join-Path $logDir "model-server.stdout.log")
        RedirectStandardError = (Join-Path $logDir "model-server.stderr.log")
        PassThru = $true
    }
    $apiProcess = Start-Process @apiStart
    @{
        pid = $apiProcess.Id
        executable = $pythonExe
        started_utc = [datetime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateDir "model-server.json")

    Wait-HttpReady -Uri "http://127.0.0.1:8090/health/ready" -Process $apiProcess -TimeoutSeconds 90 -ServiceName "RusaModels Model Server"

    Write-Host ""
    Write-Host "RUSA-CPU-16 services are ready."
    Write-Host "Model Server: http://127.0.0.1:8090"
    Write-Host "OpenAI API:   http://127.0.0.1:8090/openai/v1"
    Write-Host "llama.cpp:    http://127.0.0.1:8081 (loopback only)"
    Write-Host "Physical cores: $physicalCores | Logical processors: $logicalProcessors | Generation threads: $threads"
Write-Host ("Windows-reported CPU features: {0}" -f ($availableCpuFeatures -join ", "))
    Write-Host "Total RAM: $memoryGiB GiB | Free RAM at startup: $freeMemoryGiB GiB"
    Write-Host "Logs: $logDir"
}
catch {
    if ($apiProcess -and -not $apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($llamaProcess -and -not $llamaProcess.HasExited) { Stop-Process -Id $llamaProcess.Id -Force -ErrorAction SilentlyContinue }
    Remove-Item -LiteralPath (Join-Path $stateDir "model-server.json") -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $stateDir "llama-server.json") -Force -ErrorAction SilentlyContinue
    throw
}
finally {
    Pop-Location
}
