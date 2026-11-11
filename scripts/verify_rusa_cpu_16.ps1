[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://127.0.0.1:8090",
    [ValidateRange(1, 20)]
    [int]$SequentialRequests = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:RUSAMODELS_API_TOKEN)) {
    throw "Set RUSAMODELS_API_TOKEN to the dedicated role=inference bearer token."
}

$headers = @{ Authorization = "Bearer $env:RUSAMODELS_API_TOKEN" }
$freeBefore = [math]::Round(((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB) / 1GB, 2)
if ($freeBefore -lt 2) { throw "Only $freeBefore GiB RAM is free before acceptance testing; minimum is 2 GiB." }

$null = Invoke-RestMethod -Uri "$ApiBaseUrl/health/ready" -Method Get -TimeoutSec 10
$models = Invoke-RestMethod -Uri "$ApiBaseUrl/openai/v1/models" -Headers $headers -Method Get -TimeoutSec 10
if (@($models.data | Where-Object { $_.id -eq "rusa-cpu-4b" }).Count -ne 1) {
    throw "The model ID rusa-cpu-4b is not exposed to the inference principal."
}

$latencies = @()
for ($index = 1; $index -le $SequentialRequests; $index++) {
    $payload = @{
        model = "rusa-cpu-4b"
        messages = @(@{ role = "user"; content = "Reply with exactly OK." })
        temperature = 0
        max_tokens = 16
    } | ConvertTo-Json -Depth 8

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri "$ApiBaseUrl/openai/v1/chat/completions" -Method Post -Headers $headers -ContentType "application/json" -Body $payload -TimeoutSec 360
    $timer.Stop()

    if (-not $response.choices -or -not $response.choices[0].message.content) {
        throw "Sequential inference request $index returned no assistant content."
    }
    $latencies += [math]::Round($timer.Elapsed.TotalSeconds, 3)
    Write-Host "Inference request $index/$SequentialRequests passed ($($latencies[-1]) s)."
}

$freeAfter = [math]::Round(((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB) / 1GB, 2)
if ($freeAfter -lt 2) { throw "Free RAM fell to $freeAfter GiB; RUSA-CPU-16 requires at least 2 GiB." }
$sorted = @($latencies | Sort-Object)
$p50 = $sorted[[int][math]::Floor(($sorted.Count - 1) * 0.5)]
$average = [math]::Round((($latencies | Measure-Object -Average).Average), 3)
$maximum = ($latencies | Measure-Object -Maximum).Maximum

[pscustomobject]@{
    Status = "PASS"
    SequentialRequests = $SequentialRequests
    AverageLatencySeconds = $average
    P50LatencySeconds = $p50
    MaxLatencySeconds = $maximum
    FreeMemoryBeforeGiB = $freeBefore
    FreeMemoryAfterGiB = $freeAfter
} | Format-List
