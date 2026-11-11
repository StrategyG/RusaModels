[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ApiBaseUrl = "http://127.0.0.1:8090",
    [ValidateRange(30, 30)]
    [int]$DurationMinutes = 30,
    [ValidateRange(30, 120)]
    [int]$RequestIntervalSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$parsedApiBaseUrl = [Uri]$ApiBaseUrl
if (
    -not $parsedApiBaseUrl.IsAbsoluteUri -or
    $parsedApiBaseUrl.Scheme -ne "http" -or
    $parsedApiBaseUrl.Host -ne "127.0.0.1" -or
    $parsedApiBaseUrl.Port -ne 8090 -or
    $parsedApiBaseUrl.AbsolutePath -ne "/" -or
    $parsedApiBaseUrl.UserInfo -or
    $parsedApiBaseUrl.Query -or
    $parsedApiBaseUrl.Fragment
) {
    throw "RUSA-CPU-16 soak acceptance must target http://127.0.0.1:8090 only."
}
$ApiBaseUrl = $parsedApiBaseUrl.GetLeftPart([System.UriPartial]::Authority)

if ([string]::IsNullOrWhiteSpace($env:RUSAMODELS_API_TOKEN)) {
    throw "Set RUSAMODELS_API_TOKEN to the dedicated role=inference bearer token."
}

function Get-RusaSoakResourceSample {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $processors = @(Get-CimInstance -ClassName Win32_Processor)
    $cpuValues = @($processors | Where-Object { $null -ne $_.LoadPercentage } | ForEach-Object { [int]$_.LoadPercentage })
    $cpuLoad = $null
    if ($cpuValues.Count -gt 0) {
        $cpuLoad = [int][math]::Round((($cpuValues | Measure-Object -Average).Average), 0)
    }
    [pscustomobject]@{
        FreeMemoryGiB = [math]::Round((($os.FreePhysicalMemory * 1KB) / 1GB), 2)
        CpuLoadPercent = $cpuLoad
    }
}

function Add-RusaSoakSample {
    $sample = Get-RusaSoakResourceSample
    if ($sample.FreeMemoryGiB -lt 2) {
        throw "Free RAM fell below 2 GiB during the 30-minute soak: $($sample.FreeMemoryGiB) GiB."
    }
    $script:freeMemorySamples += $sample.FreeMemoryGiB
    if ($null -ne $sample.CpuLoadPercent) {
        $script:cpuSamples += $sample.CpuLoadPercent
    }
}

function Assert-RusaSoakProcess {
    param([string]$ServiceName)

    $statePath = Join-Path $script:stateDir ($ServiceName + ".json")
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        throw "Missing $ServiceName process record; the runtime may have restarted."
    }
    $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
    $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        throw "$ServiceName process PID $($state.pid) is no longer running."
    }
    $actualPath = $null
    try {
        $actualPath = $process.Path
    }
    catch {
        throw "Cannot verify executable path for $ServiceName PID $($state.pid)."
    }
    if ([string]::IsNullOrWhiteSpace([string]$actualPath)) {
        throw "Cannot verify executable path for $ServiceName PID $($state.pid)."
    }
    $expectedPath = [System.IO.Path]::GetFullPath([string]$state.executable)
    $actualFullPath = [System.IO.Path]::GetFullPath([string]$actualPath)
    if ($actualFullPath -ine $expectedPath) {
        throw "$ServiceName PID $($state.pid) no longer matches its recorded executable."
    }
    return [int]$state.pid
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
Push-Location $RepoRoot
$script:stateDir = Join-Path $RepoRoot (Join-Path ".local" "rusa-cpu-16")
$script:freeMemorySamples = @()
$script:cpuSamples = @()
$client = $null
$inferenceRequests = 0
$requestLatencies = @()
$llamaPid = $null
$modelServerPid = $null

try {
    $initialSample = Get-RusaSoakResourceSample
    if ($initialSample.FreeMemoryGiB -lt 2) {
        throw "Only $($initialSample.FreeMemoryGiB) GiB RAM is free before soak acceptance; minimum is 2 GiB."
    }
    $script:freeMemorySamples += $initialSample.FreeMemoryGiB
    if ($null -ne $initialSample.CpuLoadPercent) {
        $script:cpuSamples += $initialSample.CpuLoadPercent
    }
    $totalMemoryGiB = [math]::Round(((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 2)

    $llamaPid = Assert-RusaSoakProcess -ServiceName "llama-server"
    $modelServerPid = Assert-RusaSoakProcess -ServiceName "model-server"
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -Method Get -TimeoutSec 10
    $null = Invoke-RestMethod -Uri "$ApiBaseUrl/health/ready" -Method Get -TimeoutSec 10
    $models = Invoke-RestMethod -Uri "$ApiBaseUrl/openai/v1/models" -Headers @{ Authorization = "Bearer $env:RUSAMODELS_API_TOKEN" } -Method Get -TimeoutSec 10
    if (@($models.data | Where-Object { $_.id -eq "rusaai-edge-1.54b" }).Count -ne 1) {
        throw "The model ID rusaai-edge-1.54b is not exposed to the inference principal."
    }

    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(6)
    $sessionTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $deadline = [datetime]::UtcNow.AddMinutes($DurationMinutes)
    $nextRequest = [datetime]::UtcNow

    while ([datetime]::UtcNow -lt $deadline) {
        $currentLlamaPid = Assert-RusaSoakProcess -ServiceName "llama-server"
        $currentModelServerPid = Assert-RusaSoakProcess -ServiceName "model-server"
        if ($currentLlamaPid -ne $llamaPid -or $currentModelServerPid -ne $modelServerPid) {
            throw "A runtime process ID changed during the stable-session acceptance window."
        }

        if ([datetime]::UtcNow -ge $nextRequest) {
            $payload = @{
                model = "rusaai-edge-1.54b"
                messages = @(@{ role = "user"; content = "Reply with exactly OK." })
                temperature = 0
                max_tokens = 16
                stream = $false
            } | ConvertTo-Json -Depth 8

            $request = [System.Net.Http.HttpRequestMessage]::new(
                [System.Net.Http.HttpMethod]::Post,
                "$ApiBaseUrl/openai/v1/chat/completions"
            )
            $request.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new(
                "Bearer",
                $env:RUSAMODELS_API_TOKEN
            )
            $request.Content = [System.Net.Http.StringContent]::new(
                $payload,
                [System.Text.Encoding]::UTF8,
                "application/json"
            )

            $requestTimer = [System.Diagnostics.Stopwatch]::StartNew()
            $response = $null
            try {
                $sendTask = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
                while (-not $sendTask.IsCompleted) {
                    Add-RusaSoakSample
                    if ($requestTimer.Elapsed.TotalSeconds -gt 360) {
                        $client.CancelPendingRequests()
                        throw "Soak inference request exceeded 360 seconds."
                    }
                    Start-Sleep -Milliseconds 500
                }
                $response = $sendTask.GetAwaiter().GetResult()
                if (-not $response.IsSuccessStatusCode) {
                    throw "Soak inference returned HTTP $([int]$response.StatusCode)."
                }

                $bodyTask = $response.Content.ReadAsStringAsync()
                while (-not $bodyTask.IsCompleted) {
                    Add-RusaSoakSample
                    if ($requestTimer.Elapsed.TotalSeconds -gt 360) {
                        $client.CancelPendingRequests()
                        throw "Soak inference response body exceeded 360 seconds."
                    }
                    Start-Sleep -Milliseconds 500
                }
                $responseObject = ConvertFrom-Json -InputObject $bodyTask.GetAwaiter().GetResult()
                $choicesProperty = $responseObject.PSObject.Properties["choices"]
                if ($null -eq $choicesProperty -or @($choicesProperty.Value).Count -lt 1) {
                    throw "Soak inference response did not contain a completion choice."
                }
                $messageProperty = $choicesProperty.Value[0].PSObject.Properties["message"]
                if ($null -eq $messageProperty -or $null -eq $messageProperty.Value) {
                    throw "Soak inference response did not contain an assistant message."
                }
                $contentProperty = $messageProperty.Value.PSObject.Properties["content"]
                if ($null -eq $contentProperty -or [string]::IsNullOrWhiteSpace([string]$contentProperty.Value)) {
                    throw "Soak inference returned an empty assistant response."
                }
                $requestTimer.Stop()
                $requestLatencies += [math]::Round($requestTimer.Elapsed.TotalSeconds, 3)
                $inferenceRequests += 1
                Write-Host ("Soak inference {0} passed | latency {1}s | elapsed {2}m" -f $inferenceRequests, $requestLatencies[-1], [math]::Round($sessionTimer.Elapsed.TotalMinutes, 1))
            }
            finally {
                if ($response) { $response.Dispose() }
                $request.Dispose()
            }
            $nextRequest = [datetime]::UtcNow.AddSeconds($RequestIntervalSeconds)
        }
        else {
            Add-RusaSoakSample
            $null = Invoke-RestMethod -Uri "$ApiBaseUrl/health/ready" -Method Get -TimeoutSec 10
            Start-Sleep -Seconds 5
        }
    }

    $sessionTimer.Stop()
    Add-RusaSoakSample
    $null = Invoke-RestMethod -Uri "$ApiBaseUrl/health/ready" -Method Get -TimeoutSec 10
    $finalLlamaPid = Assert-RusaSoakProcess -ServiceName "llama-server"
    $finalModelServerPid = Assert-RusaSoakProcess -ServiceName "model-server"
    if ($finalLlamaPid -ne $llamaPid -or $finalModelServerPid -ne $modelServerPid) {
        throw "A runtime process changed before the stable-session acceptance completed."
    }

    $minimumFreeMemoryGiB = ($script:freeMemorySamples | Measure-Object -Minimum).Minimum
    $cpuAverage = $null
    $cpuMaximum = $null
    if ($script:cpuSamples.Count -gt 0) {
        $cpuAverage = [math]::Round((($script:cpuSamples | Measure-Object -Average).Average), 1)
        $cpuMaximum = ($script:cpuSamples | Measure-Object -Maximum).Maximum
    }
    $averageRequestLatency = $null
    $maximumRequestLatency = $null
    if ($requestLatencies.Count -gt 0) {
        $averageRequestLatency = [math]::Round((($requestLatencies | Measure-Object -Average).Average), 3)
        $maximumRequestLatency = ($requestLatencies | Measure-Object -Maximum).Maximum
    }

    $report = [pscustomobject]@{
        Status = "PASS"
        Profile = "RUSA-CPU-16-v2.0"
        ModelId = "rusaai-edge-1.54b"
        StableSessionMinutes = $DurationMinutes
        ActualElapsedMinutes = [math]::Round($sessionTimer.Elapsed.TotalMinutes, 2)
        RequestIntervalSeconds = $RequestIntervalSeconds
        InferenceRequests = $inferenceRequests
        AverageRequestLatencySeconds = $averageRequestLatency
        MaximumRequestLatencySeconds = $maximumRequestLatency
        MinimumSampledFreeMemoryGiB = $minimumFreeMemoryGiB
        MaximumObservedSystemMemoryUsedGiB = [math]::Round(($totalMemoryGiB - $minimumFreeMemoryGiB), 2)
        AverageSampledCpuLoadPercent = $cpuAverage
        MaximumSampledCpuLoadPercent = $cpuMaximum
        MemorySampleCount = $script:freeMemorySamples.Count
        MemorySampleIntervalIdleSeconds = 5
        MemorySampleIntervalDuringInferenceMilliseconds = 500
        ProcessIds = @{ LlamaServer = $llamaPid; ModelServer = $modelServerPid }
        RestartDetected = $false
        ThermalTelemetry = "Not collected by this script; inspect device telemetry separately."
        SwapTelemetry = "Not collected by this script; inspect operating-system telemetry separately."
        QueueTelemetryOwner = "Rusaai"
        CapturedUtc = [datetime]::UtcNow.ToString("o")
    }

    $reportDir = Join-Path $RepoRoot (Join-Path ".local" "logs")
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    $reportPath = Join-Path $reportDir ("rusa-cpu-16-soak-{0}.json" -f [datetime]::UtcNow.ToString("yyyyMMdd-HHmmss"))
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath
    $report | Format-List
    Write-Host "Stable-session report: $reportPath"
}
finally {
    if ($client) { $client.Dispose() }
    Pop-Location
}
