[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://127.0.0.1:8090",
    [ValidateRange(20, 20)]
    [int]$SequentialRequests = 20
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
    throw "RUSA-CPU-16 acceptance must target http://127.0.0.1:8090 only."
}
$ApiBaseUrl = $parsedApiBaseUrl.GetLeftPart([System.UriPartial]::Authority)

function Get-RusaResourceSample {
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

function Add-RusaResourceSample {
    $sample = Get-RusaResourceSample
    if ($sample.FreeMemoryGiB -lt 2) {
        throw "Free RAM fell to $($sample.FreeMemoryGiB) GiB during inference; RUSA-CPU-16 requires at least 2 GiB."
    }
    if ($null -eq $script:minimumFreeMemoryGiB -or $sample.FreeMemoryGiB -lt $script:minimumFreeMemoryGiB) {
        $script:minimumFreeMemoryGiB = $sample.FreeMemoryGiB
    }
    if ($null -ne $sample.CpuLoadPercent) {
        $script:cpuSamples += $sample.CpuLoadPercent
    }
}

function Get-LlamaMetricSnapshot {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8081/metrics" -Method Get -TimeoutSec 10 -UseBasicParsing
    $metrics = @{}
    foreach ($line in ($response.Content -split "\r?\n")) {
        if ($line -match '^llamacpp:(?<name>tokens_predicted_total|tokens_predicted_seconds_total)(?:\{[^}]*\})?\s+(?<value>[0-9.eE+-]+)$') {
            $metrics[$Matches.name] = [double]$Matches.value
        }
    }
    foreach ($required in @("tokens_predicted_total", "tokens_predicted_seconds_total")) {
        if (-not $metrics.ContainsKey($required)) {
            throw "llama.cpp /metrics did not expose llamacpp:$required."
        }
    }
    return $metrics
}

if ([string]::IsNullOrWhiteSpace($env:RUSAMODELS_API_TOKEN)) {
    throw "Set RUSAMODELS_API_TOKEN to the dedicated role=inference bearer token."
}

$initialSample = Get-RusaResourceSample
if ($initialSample.FreeMemoryGiB -lt 2) {
    throw "Only $($initialSample.FreeMemoryGiB) GiB RAM is free before acceptance testing; minimum is 2 GiB."
}
$script:minimumFreeMemoryGiB = $initialSample.FreeMemoryGiB
$script:cpuSamples = @()
if ($null -ne $initialSample.CpuLoadPercent) {
    $script:cpuSamples += $initialSample.CpuLoadPercent
}
$totalMemoryGiB = [math]::Round(((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 2)

$null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -Method Get -TimeoutSec 10
$null = Invoke-RestMethod -Uri "$ApiBaseUrl/health/ready" -Method Get -TimeoutSec 10
$models = Invoke-RestMethod -Uri "$ApiBaseUrl/openai/v1/models" -Headers @{ Authorization = "Bearer $env:RUSAMODELS_API_TOKEN" } -Method Get -TimeoutSec 10
if (@($models.data | Where-Object { $_.id -eq "rusa-cpu-4b" }).Count -ne 1) {
    throw "The model ID rusa-cpu-4b is not exposed to the inference principal."
}

Add-Type -AssemblyName System.Net.Http
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(6)

$latencies = @()
$ttfts = @()
$tokensPerSecond = @()

try {
    for ($index = 1; $index -le $SequentialRequests; $index++) {
        $beforeMetrics = Get-LlamaMetricSnapshot
        $payload = @{
            model = "rusa-cpu-4b"
            messages = @(@{ role = "user"; content = "Reply with exactly OK." })
            temperature = 0
            max_tokens = 16
            stream = $true
        } | ConvertTo-Json -Depth 8

        $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$ApiBaseUrl/openai/v1/chat/completions")
        $request.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $env:RUSAMODELS_API_TOKEN)
        $request.Content = [System.Net.Http.StringContent]::new($payload, [System.Text.Encoding]::UTF8, "application/json")

        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        $response = $null
        $reader = $null
        try {
            $sendTask = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
            while (-not $sendTask.IsCompleted) {
                Add-RusaResourceSample
                if ($timer.Elapsed.TotalSeconds -gt 360) {
                    $client.CancelPendingRequests()
                    throw "Inference request $index exceeded the 360-second request limit."
                }
                Start-Sleep -Milliseconds 500
            }
            $response = $sendTask.GetAwaiter().GetResult()
            if (-not $response.IsSuccessStatusCode) {
                throw "Inference request $index returned HTTP $([int]$response.StatusCode)."
            }

            $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $reader = [System.IO.StreamReader]::new($stream)
            $firstTokenSeconds = $null
            $assistantContent = [System.Text.StringBuilder]::new()
            $sawDone = $false

            while ($true) {
                $lineTask = $reader.ReadLineAsync()
                while (-not $lineTask.IsCompleted) {
                    Add-RusaResourceSample
                    if ($timer.Elapsed.TotalSeconds -gt 360) {
                        $client.CancelPendingRequests()
                        throw "Inference request $index exceeded the 360-second request limit."
                    }
                    Start-Sleep -Milliseconds 500
                }
                $line = $lineTask.GetAwaiter().GetResult()
                if ($null -eq $line) { break }
                if (-not $line.StartsWith("data:")) { continue }

                $data = $line.Substring(5).Trim()
                if ($data -eq "[DONE]") {
                    $sawDone = $true
                    break
                }
                if ([string]::IsNullOrWhiteSpace($data)) { continue }

                try {
                    $chunk = ConvertFrom-Json -InputObject $data -ErrorAction Stop
                }
                catch {
                    continue
                }
                if (-not $chunk.choices -or $null -eq $chunk.choices[0].delta) { continue }

                $delta = $chunk.choices[0].delta
                if ($null -ne $delta.content -and [string]$delta.content -ne "") {
                    [void]$assistantContent.Append([string]$delta.content)
                    if ($null -eq $firstTokenSeconds) {
                        $firstTokenSeconds = $timer.Elapsed.TotalSeconds
                    }
                }
                if ($null -ne $delta.reasoning_content -and [string]$delta.reasoning_content -ne "" -and $null -eq $firstTokenSeconds) {
                    $firstTokenSeconds = $timer.Elapsed.TotalSeconds
                }
            }

            $timer.Stop()
            if (-not $sawDone) { throw "Inference request $index ended without the SSE [DONE] event." }
            if ($assistantContent.Length -eq 0) { throw "Inference request $index returned no assistant content." }
            if ($null -eq $firstTokenSeconds) { throw "Inference request $index did not emit a measurable first token." }
        }
        finally {
            if ($reader) { $reader.Dispose() }
            if ($response) { $response.Dispose() }
            $request.Dispose()
        }

        Add-RusaResourceSample
        $afterMetrics = Get-LlamaMetricSnapshot
        $tokenDelta = $afterMetrics["tokens_predicted_total"] - $beforeMetrics["tokens_predicted_total"]
        $generationSecondsDelta = $afterMetrics["tokens_predicted_seconds_total"] - $beforeMetrics["tokens_predicted_seconds_total"]
        if ($tokenDelta -le 0 -or $generationSecondsDelta -le 0) {
            throw "llama.cpp generation counters did not advance for request $index."
        }

        $requestTps = [math]::Round(($tokenDelta / $generationSecondsDelta), 2)
        $latencies += [math]::Round($timer.Elapsed.TotalSeconds, 3)
        $ttfts += [math]::Round($firstTokenSeconds, 3)
        $tokensPerSecond += $requestTps
        Write-Host ("Inference request {0}/{1} passed | TTFT {2}s | total {3}s | {4} tokens/s" -f $index, $SequentialRequests, $ttfts[-1], $latencies[-1], $requestTps)
    }
}
finally {
    $client.Dispose()
}

$finalSample = Get-RusaResourceSample
if ($finalSample.FreeMemoryGiB -lt 2) {
    throw "Free RAM fell to $($finalSample.FreeMemoryGiB) GiB after inference; minimum is 2 GiB."
}
if ($finalSample.FreeMemoryGiB -lt $script:minimumFreeMemoryGiB) {
    $script:minimumFreeMemoryGiB = $finalSample.FreeMemoryGiB
}
if ($null -ne $finalSample.CpuLoadPercent) {
    $script:cpuSamples += $finalSample.CpuLoadPercent
}

$sortedLatency = @($latencies | Sort-Object)
$sortedTtft = @($ttfts | Sort-Object)
$sortedTps = @($tokensPerSecond | Sort-Object)
$cpuAverage = $null
$cpuMaximum = $null
if ($script:cpuSamples.Count -gt 0) {
    $cpuAverage = [math]::Round((($script:cpuSamples | Measure-Object -Average).Average), 1)
    $cpuMaximum = ($script:cpuSamples | Measure-Object -Maximum).Maximum
}
$report = [pscustomobject]@{
    Status = "PASS"
    Profile = "RUSA-CPU-16"
    SequentialRequests = $SequentialRequests
    AverageLatencySeconds = [math]::Round((($latencies | Measure-Object -Average).Average), 3)
    P50LatencySeconds = $sortedLatency[[int][math]::Floor(($sortedLatency.Count - 1) * 0.5)]
    MaximumLatencySeconds = ($latencies | Measure-Object -Maximum).Maximum
    AverageTTFTSeconds = [math]::Round((($ttfts | Measure-Object -Average).Average), 3)
    P50TTFTSeconds = $sortedTtft[[int][math]::Floor(($sortedTtft.Count - 1) * 0.5)]
    AverageGenerationTokensPerSecond = [math]::Round((($tokensPerSecond | Measure-Object -Average).Average), 2)
    P50GenerationTokensPerSecond = $sortedTps[[int][math]::Floor(($sortedTps.Count - 1) * 0.5)]
    MinimumSampledFreeMemoryGiB = $script:minimumFreeMemoryGiB
    EstimatedPeakSystemMemoryUsedGiB = [math]::Round(($totalMemoryGiB - $script:minimumFreeMemoryGiB), 2)
    AverageSampledCpuLoadPercent = $cpuAverage
    MaximumSampledCpuLoadPercent = $cpuMaximum
    CpuSampleSource = "Win32_Processor.LoadPercentage"
    MemorySampleIntervalMilliseconds = 500
    QueueWaitTime = "Collect separately from Rusaai scheduler telemetry"
}

$reportDir = Join-Path (Get-Location).Path ".local\logs"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$reportPath = Join-Path $reportDir ("rusa-cpu-16-acceptance-{0}.json" -f [datetime]::UtcNow.ToString("yyyyMMdd-HHmmss"))
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath
$report | Format-List
Write-Host "Acceptance report: $reportPath"
