[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://127.0.0.1:8090"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$parsedApiBaseUrl = [Uri]$ApiBaseUrl
if (-not $parsedApiBaseUrl.IsAbsoluteUri -or $parsedApiBaseUrl.Scheme -ne "http" -or $parsedApiBaseUrl.Host -ne "127.0.0.1" -or $parsedApiBaseUrl.Port -ne 8090 -or $parsedApiBaseUrl.AbsolutePath -ne "/" -or $parsedApiBaseUrl.UserInfo -or $parsedApiBaseUrl.Query -or $parsedApiBaseUrl.Fragment) {
    throw "RUSA-CPU-16 context acceptance must target http://127.0.0.1:8090 only."
}
$ApiBaseUrl = $parsedApiBaseUrl.GetLeftPart([System.UriPartial]::Authority)

if ($env:RUSAMODELS_CONTEXT_SIZE -ne "8192" -or $env:RUSAMODELS_CONTEXT_8K_TEST_MODE -ne "true") {
    throw "Set RUSAMODELS_CONTEXT_SIZE=8192 and RUSAMODELS_CONTEXT_8K_TEST_MODE=true for this acceptance test."
}
if ([string]::IsNullOrWhiteSpace($env:RUSAMODELS_API_TOKEN)) {
    throw "Set RUSAMODELS_API_TOKEN to the dedicated role=inference bearer token."
}

$script:freeMemorySamples = @()
function Add-RusaFreeMemorySample {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $freeGiB = [math]::Round((($os.FreePhysicalMemory * 1KB) / 1GB), 2)
    if ($freeGiB -lt 2) {
        throw "Free RAM fell below 2 GiB during 8K context acceptance: $freeGiB GiB."
    }
    $script:freeMemorySamples += $freeGiB
    return $freeGiB
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

$client = $null
$request = $null
$response = $null
try {
    $initialFreeMemoryGiB = Add-RusaFreeMemorySample
    $totalMemoryGiB = [math]::Round(((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 2)

    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -Method Get -TimeoutSec 10
    $models = Invoke-RestMethod -Uri "$ApiBaseUrl/openai/v1/models" -Headers @{ Authorization = "Bearer $env:RUSAMODELS_API_TOKEN" } -Method Get -TimeoutSec 10
    if (@($models.data | Where-Object { $_.id -eq "rusaai-edge-1.54b" }).Count -ne 1) {
        throw "The model ID rusaai-edge-1.54b is not exposed to the inference principal."
    }

    $seed = "RusaAi local context acceptance sequence 0123456789. "
    $low = 1
    $high = 4096
    $bestRepeat = 0
    $bestPromptTokens = 0
    $bestContent = ""
    while ($low -le $high) {
        $middle = [int][math]::Floor(($low + $high) / 2)
        $candidate = $seed * $middle
        $candidateTokens = Get-LlamaTokenCount -Content $candidate
        if ($candidateTokens -le 7600) {
            $bestRepeat = $middle
            $bestPromptTokens = $candidateTokens
            $bestContent = $candidate
            $low = $middle + 1
        }
        else {
            $high = $middle - 1
        }
    }
    if ($bestPromptTokens -lt 7400) {
        throw ("Could not construct a safe 7400 to 7600 token context probe; observed {0} tokens." -f $bestPromptTokens)
    }

    $payload = @{
        model = "rusaai-edge-1.54b"
        messages = @(@{ role = "user"; content = $bestContent })
        temperature = 0
        max_tokens = 64
        stream = $false
    } | ConvertTo-Json -Depth 8

    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(11)
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
    $sendTask = $client.SendAsync(
        $request,
        [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
    )
    while (-not $sendTask.IsCompleted) {
        Add-RusaFreeMemorySample | Out-Null
        if ($timer.Elapsed.TotalSeconds -gt 600) {
            $client.CancelPendingRequests()
            throw "8K context inference exceeded the 600-second acceptance timeout."
        }
        Start-Sleep -Milliseconds 500
    }
    $response = $sendTask.GetAwaiter().GetResult()
    if (-not $response.IsSuccessStatusCode) {
        $errorBody = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        throw "8K context inference returned HTTP $([int]$response.StatusCode): $errorBody"
    }

    $bodyTask = $response.Content.ReadAsStringAsync()
    while (-not $bodyTask.IsCompleted) {
        Add-RusaFreeMemorySample | Out-Null
        if ($timer.Elapsed.TotalSeconds -gt 600) {
            $client.CancelPendingRequests()
            throw "8K context response body exceeded the 600-second acceptance timeout."
        }
        Start-Sleep -Milliseconds 500
    }
    $responseBody = $bodyTask.GetAwaiter().GetResult()
    $responseObject = ConvertFrom-Json -InputObject $responseBody
    $timer.Stop()

    if ($null -eq $responseObject.usage -or $null -eq $responseObject.usage.prompt_tokens -or $null -eq $responseObject.usage.completion_tokens) {
        throw "The inference response did not provide prompt_tokens and completion_tokens usage."
    }
    $actualPromptTokens = [int]$responseObject.usage.prompt_tokens
    $completionTokens = [int]$responseObject.usage.completion_tokens
    $totalTokens = $actualPromptTokens + $completionTokens
    if ($null -ne $responseObject.usage.total_tokens) {
        $totalTokens = [int]$responseObject.usage.total_tokens
    }
    if ($actualPromptTokens -lt 7400 -or $totalTokens -gt 8192) {
        throw "The response did not validate the 8K context boundary: prompt=$actualPromptTokens total=$totalTokens."
    }

    $finalFreeMemoryGiB = Add-RusaFreeMemorySample
    $minimumFreeMemoryGiB = ($script:freeMemorySamples | Measure-Object -Minimum).Minimum
    $report = [pscustomobject]@{
        Status = "PASS"
        Profile = "RUSA-CPU-16-v2.0"
        ModelId = "rusaai-edge-1.54b"
        ContextLimitTokens = 8192
        TokenizerProbeTokens = $bestPromptTokens
        ActualPromptTokens = $actualPromptTokens
        CompletionTokens = $completionTokens
        TotalTokens = $totalTokens
        RequestLatencySeconds = [math]::Round($timer.Elapsed.TotalSeconds, 3)
        InitialFreeMemoryGiB = $initialFreeMemoryGiB
        FinalFreeMemoryGiB = $finalFreeMemoryGiB
        MinimumSampledFreeMemoryGiB = $minimumFreeMemoryGiB
        TotalPhysicalMemoryGiB = $totalMemoryGiB
        MemorySampleCount = $script:freeMemorySamples.Count
        MemorySampleIntervalMilliseconds = 500
        MemorySampleSource = "Win32_OperatingSystem.FreePhysicalMemory"
        CapturedUtc = [datetime]::UtcNow.ToString("o")
        ManifestUpdateRequired = "Set benchmark.context_8192_status to PASS only after reviewing this report."
    }

    $RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
    $reportDir = Join-Path $RepoRoot (Join-Path ".local" "logs")
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    $reportPath = Join-Path $reportDir ("rusa-cpu-16-context-8k-{0}.json" -f [datetime]::UtcNow.ToString("yyyyMMdd-HHmmss"))
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath
    $report | Format-List
    Write-Host "8K context acceptance report: $reportPath"
}
finally {
    if ($request) { $request.Dispose() }
    if ($response) { $response.Dispose() }
    if ($client) { $client.Dispose() }
}
