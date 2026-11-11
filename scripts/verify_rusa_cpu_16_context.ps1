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

function Get-FreeMemoryGiB {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    return [math]::Round((($os.FreePhysicalMemory * 1KB) / 1GB), 2)
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

$initialFreeMemoryGiB = Get-FreeMemoryGiB
if ($initialFreeMemoryGiB -lt 2) {
    throw "Only $initialFreeMemoryGiB GiB RAM is free before the 8K context test; minimum is 2 GiB."
}
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
    throw "Could not construct a safe 7400-7600 token context probe; observed $bestPromptTokens tokens."
}

$payload = @{
    model = "rusaai-edge-1.54b"
    messages = @(@{ role = "user"; content = $bestContent })
    temperature = 0
    max_tokens = 64
    stream = $false
} | ConvertTo-Json -Depth 8

$timer = [System.Diagnostics.Stopwatch]::StartNew()
$response = Invoke-RestMethod -Uri "$ApiBaseUrl/openai/v1/chat/completions" -Headers @{ Authorization = "Bearer $env:RUSAMODELS_API_TOKEN" } -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 600
$timer.Stop()

if ($null -eq $response.usage -or $null -eq $response.usage.prompt_tokens -or $null -eq $response.usage.completion_tokens) {
    throw "The inference response did not provide prompt_tokens and completion_tokens usage."
}
$actualPromptTokens = [int]$response.usage.prompt_tokens
$completionTokens = [int]$response.usage.completion_tokens
$totalTokens = $actualPromptTokens + $completionTokens
if ($null -ne $response.usage.total_tokens) {
    $totalTokens = [int]$response.usage.total_tokens
}
if ($actualPromptTokens -lt 7400 -or $totalTokens -gt 8192) {
    throw "The response did not validate the 8K context boundary: prompt=$actualPromptTokens total=$totalTokens."
}

$finalFreeMemoryGiB = Get-FreeMemoryGiB
if ($finalFreeMemoryGiB -lt 2) {
    throw "Free RAM fell to $finalFreeMemoryGiB GiB after the 8K context test; minimum is 2 GiB."
}
$minimumSampledFreeMemoryGiB = [math]::Min($initialFreeMemoryGiB, $finalFreeMemoryGiB)

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
    MinimumSampledFreeMemoryGiB = $minimumSampledFreeMemoryGiB
    TotalPhysicalMemoryGiB = $totalMemoryGiB
    MemorySamples = 2
    MemorySamplingNote = "Before and after request only; not continuous peak-RAM telemetry."
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
