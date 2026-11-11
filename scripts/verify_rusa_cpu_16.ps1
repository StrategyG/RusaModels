[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
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

function Get-RusaApprovedRuntimeModelId {
    param([string]$Root)

    $manifestPath = $env:RUSAMODELS_MODEL_MANIFEST
    if ([string]::IsNullOrWhiteSpace($manifestPath)) {
        $manifestPath = Join-Path $Root "models/rusaai-edge/manifests/0.1.0-design.json"
    }
    elseif (-not [System.IO.Path]::IsPathRooted($manifestPath)) {
        $manifestPath = Join-Path $Root $manifestPath
    }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Model manifest is missing: $manifestPath"
    }

    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($manifest.status -ne "READY_FOR_DEVICE_VALIDATION") {
        throw "Acceptance requires manifest status READY_FOR_DEVICE_VALIDATION."
    }
    if ($manifest.model_id -ne $manifest.identity_alignment.architecture_reference_model_id) {
        throw "Manifest model_id must remain the Architecture v2.0 reference identity."
    }
    $runtimeModelId = [string]$manifest.identity_alignment.runtime_model_id
    if ($manifest.identity_alignment.runtime_model_id_decision -ne "APPROVED" -or
        [string]::IsNullOrWhiteSpace($runtimeModelId)) {
        throw "Acceptance blocked: ADR-TRAIN-001 must approve and record identity_alignment.runtime_model_id."
    }
    if (-not [string]::IsNullOrWhiteSpace($env:RUSAMODELS_INFERENCE_MODEL_ID) -and
        $env:RUSAMODELS_INFERENCE_MODEL_ID -ne $runtimeModelId) {
        throw "RUSAMODELS_INFERENCE_MODEL_ID does not match the approved manifest runtime_model_id."
    }
    return $runtimeModelId
}


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

function Get-LlamaTokenCount {
    param([string]$Content)
    $body = @{ content = $Content } | ConvertTo-Json -Compress -Depth 3
    $result = Invoke-RestMethod -Uri "http://127.0.0.1:8081/tokenize" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 120
    if ($null -eq $result.tokens) {
        throw "llama.cpp /tokenize did not return a token array."
    }
    return @($result.tokens).Count
}

function New-RusaPromptContent {
    param([int]$TargetContentTokens)

    $seed = "RusaAi CPU acceptance prompt sequence 0123456789. "
    $low = 1
    $high = $TargetContentTokens * 4
    $bestContent = ""
    $bestTokens = 0
    while ($low -le $high) {
        $middle = [int][math]::Floor(($low + $high) / 2)
        $candidate = $seed * $middle
        $candidateTokens = Get-LlamaTokenCount -Content $candidate
        if ($candidateTokens -le $TargetContentTokens) {
            $bestContent = $candidate
            $bestTokens = $candidateTokens
            $low = $middle + 1
        }
        else {
            $high = $middle - 1
        }
    }
    if ($bestTokens -lt [math]::Floor($TargetContentTokens * 0.95)) {
        throw "Could not construct the requested prompt size; target=$TargetContentTokens actual=$bestTokens."
    }
    return [pscustomobject]@{
        Content = $bestContent
        Tokens = $bestTokens
    }
}

function Get-LlamaMetricSnapshot {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8081/metrics" -Method Get -TimeoutSec 10 -UseBasicParsing
    $metrics = @{}
    foreach ($line in ($response.Content -split "\r?\n")) {
        if ($line -match '^llamacpp:(?<name>tokens_predicted_total|tokens_predicted_seconds_total|requests_processing|requests_deferred)(?:\{[^}]*\})?\s+(?<value>[0-9.eE+-]+)$') {
            $metrics[$Matches.name] = [double]$Matches.value
        }
    }
    foreach ($required in @("tokens_predicted_total", "tokens_predicted_seconds_total", "requests_processing", "requests_deferred")) {
        if (-not $metrics.ContainsKey($required)) {
            throw "llama.cpp /metrics did not expose llamacpp:$required."
        }
    }
    return $metrics
}

function Get-RusaMedian {
    param(
        [object[]]$Values,
        [int]$Digits = 3
    )
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) {
        throw "Cannot calculate a median from an empty sample."
    }
    $middle = [int][math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 0) {
        return [math]::Round((([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2), $Digits)
    }
    return [math]::Round([double]$sorted[$middle], $Digits)
}
if ([string]::IsNullOrWhiteSpace($env:RUSAMODELS_API_TOKEN)) {
    throw "Set RUSAMODELS_API_TOKEN to the dedicated role=inference bearer token."
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
Push-Location $RepoRoot
$client = $null
try {
    $runtimeModelId = Get-RusaApprovedRuntimeModelId -Root $RepoRoot
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
    if (@($models.data | Where-Object { $_.id -eq $runtimeModelId }).Count -ne 1) {
        throw "The model ID Approved runtime model ID $runtimeModelId is not exposed to the inference principal."
    }

    $promptTargets = [ordered]@{
        "128" = 120
        "1024" = 1008
        "4096" = 4032
    }
    $promptProfiles = @{}
    foreach ($target in $promptTargets.Keys) {
        $promptProfiles[$target] = New-RusaPromptContent -TargetContentTokens $promptTargets[$target]
    }

    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(6)

    $profileNames = @("128", "1024", "4096")
    $latencies = @()
    $ttfts = @()
    $tokensPerSecond = @()
    $latencyByPrompt = @{ "128" = @(); "1024" = @(); "4096" = @() }
    $ttftByPrompt = @{ "128" = @(); "1024" = @(); "4096" = @() }
    $tpsByPrompt = @{ "128" = @(); "1024" = @(); "4096" = @() }

    for ($index = 1; $index -le $SequentialRequests; $index++) {
        $profileName = $profileNames[($index - 1) % $profileNames.Count]
        $promptProfile = $promptProfiles[$profileName]
        $beforeMetrics = Get-LlamaMetricSnapshot
        if ($beforeMetrics["requests_processing"] -ne 0 -or $beforeMetrics["requests_deferred"] -ne 0) {
            throw "llama.cpp must be idle before acceptance request $index; stop other inference clients."
        }

        $payload = @{
            model = $runtimeModelId
            messages = @(@{ role = "user"; content = $promptProfile.Content })
            temperature = 0
            max_tokens = 16
            stream = $true
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
                $choicesProperty = $chunk.PSObject.Properties["choices"]
                if ($null -eq $choicesProperty -or @($choicesProperty.Value).Count -lt 1) { continue }
                $deltaProperty = $choicesProperty.Value[0].PSObject.Properties["delta"]
                if ($null -eq $deltaProperty -or $null -eq $deltaProperty.Value) { continue }

                $delta = $deltaProperty.Value
                $contentProperty = $delta.PSObject.Properties["content"]
                $reasoningProperty = $delta.PSObject.Properties["reasoning_content"]
                $deltaContent = $null
                $reasoningContent = $null
                if ($null -ne $contentProperty) { $deltaContent = $contentProperty.Value }
                if ($null -ne $reasoningProperty) { $reasoningContent = $reasoningProperty.Value }

                if ($null -ne $deltaContent -and [string]$deltaContent -ne "") {
                    [void]$assistantContent.Append([string]$deltaContent)
                    if ($null -eq $firstTokenSeconds) {
                        $firstTokenSeconds = $timer.Elapsed.TotalSeconds
                    }
                }
                if ($null -ne $reasoningContent -and [string]$reasoningContent -ne "" -and $null -eq $firstTokenSeconds) {
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
        if ($afterMetrics["requests_processing"] -ne 0 -or $afterMetrics["requests_deferred"] -ne 0) {
            throw "llama.cpp did not return to an idle state after acceptance request $index."
        }
        $tokenDelta = $afterMetrics["tokens_predicted_total"] - $beforeMetrics["tokens_predicted_total"]
        $generationSecondsDelta = $afterMetrics["tokens_predicted_seconds_total"] - $beforeMetrics["tokens_predicted_seconds_total"]
        if ($tokenDelta -le 0 -or $generationSecondsDelta -le 0) {
            throw "llama.cpp generation counters did not advance for request $index."
        }

        $requestTps = [math]::Round(($tokenDelta / $generationSecondsDelta), 2)
        $requestLatency = [math]::Round($timer.Elapsed.TotalSeconds, 3)
        $requestTtft = [math]::Round($firstTokenSeconds, 3)
        $latencies += $requestLatency
        $ttfts += $requestTtft
        $tokensPerSecond += $requestTps
        $latencyByPrompt[$profileName] += $requestLatency
        $ttftByPrompt[$profileName] += $requestTtft
        $tpsByPrompt[$profileName] += $requestTps
        Write-Host ("Inference request {0}/{1} | target prompt {2} | content tokens {3} | TTFT {4}s | total {5}s | {6} tokens/s" -f $index, $SequentialRequests, $profileName, $promptProfile.Tokens, $requestTtft, $requestLatency, $requestTps)
    }

    Add-RusaResourceSample
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

    $promptReports = [ordered]@{}
    foreach ($profileName in $profileNames) {
        $profileLatency = @($latencyByPrompt[$profileName] | Sort-Object)
        $profileTtft = @($ttftByPrompt[$profileName] | Sort-Object)
        $profileTps = @($tpsByPrompt[$profileName] | Sort-Object)
        $promptReports[$profileName] = [pscustomobject]@{
            PromptSizeClassTokens = [int]$profileName
            ContentTokens = $promptProfiles[$profileName].Tokens
            RequestCount = $latencyByPrompt[$profileName].Count
            AverageLatencySeconds = [math]::Round((($latencyByPrompt[$profileName] | Measure-Object -Average).Average), 3)
            P50LatencySeconds = (Get-RusaMedian -Values $latencyByPrompt[$profileName] -Digits 3)
            AverageTTFTSeconds = [math]::Round((($ttftByPrompt[$profileName] | Measure-Object -Average).Average), 3)
            P50TTFTSeconds = (Get-RusaMedian -Values $ttftByPrompt[$profileName] -Digits 3)
            AverageGenerationTokensPerSecond = [math]::Round((($tpsByPrompt[$profileName] | Measure-Object -Average).Average), 2)
            P50GenerationTokensPerSecond = (Get-RusaMedian -Values $tpsByPrompt[$profileName] -Digits 2)
        }
    }

    $report = [pscustomobject]@{
        Status = "PASS"
        Profile = "RUSA-CPU-16-v2.0"
        ModelId = $runtimeModelId
        SequentialRequests = $SequentialRequests
        PromptSizeClassesTokens = @(128, 1024, 4096)
        PromptBenchmarks = $promptReports
        AverageLatencySeconds = [math]::Round((($latencies | Measure-Object -Average).Average), 3)
        P50LatencySeconds = (Get-RusaMedian -Values $latencies -Digits 3)
        MaximumLatencySeconds = ($latencies | Measure-Object -Maximum).Maximum
        AverageTTFTSeconds = [math]::Round((($ttfts | Measure-Object -Average).Average), 3)
        P50TTFTSeconds = (Get-RusaMedian -Values $ttfts -Digits 3)
        AverageGenerationTokensPerSecond = [math]::Round((($tokensPerSecond | Measure-Object -Average).Average), 2)
        P50GenerationTokensPerSecond = (Get-RusaMedian -Values $tokensPerSecond -Digits 2)
        MinimumSampledFreeMemoryGiB = $script:minimumFreeMemoryGiB
        MaximumObservedSystemMemoryUsedGiB = [math]::Round(($totalMemoryGiB - $script:minimumFreeMemoryGiB), 2)
        AverageSampledCpuLoadPercent = $cpuAverage
        MaximumSampledCpuLoadPercent = $cpuMaximum
        CpuSampleSource = "Win32_Processor.LoadPercentage"
        MemorySampleIntervalMilliseconds = 500
        QueueWaitTime = "Collect separately from Rusaai scheduler telemetry"
    }

    $reportDir = Join-Path $RepoRoot (Join-Path ".local" "logs")
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    $reportPath = Join-Path $reportDir ("rusa-cpu-16-acceptance-{0}.json" -f [datetime]::UtcNow.ToString("yyyyMMdd-HHmmss"))
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath
    $report | Format-List
    Write-Host "Acceptance report: $reportPath"
}
finally {
    if ($client) {
        $client.Dispose()
    }
    Pop-Location
}
