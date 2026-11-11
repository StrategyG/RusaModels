[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$stateDir = Join-Path $RepoRoot ".local\rusa-cpu-16"

foreach ($service in @("model-server", "llama-server")) {
    $statePath = Join-Path $stateDir "$service.json"
    if (-not (Test-Path -LiteralPath $statePath)) { continue }

    $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
    $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($process) {
        $actualPath = $null
        try { $actualPath = $process.Path }
        catch { Write-Warning "Cannot verify executable path for PID $($state.pid); leaving it untouched." }
        if (-not $actualPath) {
            continue
        }
        if ($actualPath) {
            $expectedPath = [System.IO.Path]::GetFullPath([string]$state.executable)
            $actualFullPath = [System.IO.Path]::GetFullPath([string]$actualPath)
            if ($actualFullPath -ine $expectedPath) {
                throw "PID $($state.pid) no longer belongs to $service; refusing to stop it."
            }
            Stop-Process -Id ([int]$state.pid) -Force
            Write-Host "Stopped $service (PID $($state.pid))."
        }
    }
    Remove-Item -LiteralPath $statePath -Force
}
