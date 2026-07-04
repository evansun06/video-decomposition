param(
    [int]$Limit = 10,
    [int]$SleepSeconds = 120
)

$ErrorActionPreference = "Stop"

if ($Limit -lt 1) {
    throw "Limit must be at least 1."
}

if ($SleepSeconds -lt 1) {
    throw "SleepSeconds must be at least 1."
}

if ([string]::IsNullOrWhiteSpace($env:VIDEO_DB)) {
    throw "VIDEO_DB is not set."
}

if ([string]::IsNullOrWhiteSpace($env:NASOUTPUTPATH)) {
    throw "NASOUTPUTPATH is not set."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$countSubmittedScript = @'
import os
import sqlite3

with sqlite3.connect(os.environ["VIDEO_DB"]) as connection:
    remaining = connection.execute(
        "SELECT COUNT(*) FROM transcription_batches WHERE status = ?",
        ("submitted",),
    ).fetchone()[0]

print(remaining)
'@
$countSubmittedScriptPath = [System.IO.Path]::ChangeExtension(
    [System.IO.Path]::GetTempFileName(),
    ".py"
)
Set-Content -Path $countSubmittedScriptPath -Value $countSubmittedScript -Encoding UTF8

Push-Location $repoRoot
try {
    while ($true) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host ""
        Write-Host "[$timestamp] Polling up to $Limit submitted batches..."

        & python -m youtube_decompose.poll_transcription_batches `
            --db $env:VIDEO_DB `
            --output-root $env:NASOUTPUTPATH `
            --limit $Limit
        if ($LASTEXITCODE -ne 0) {
            throw "poll_transcription_batches exited with code $LASTEXITCODE."
        }

        & python status_counts.py $env:VIDEO_DB
        if ($LASTEXITCODE -ne 0) {
            throw "status_counts.py exited with code $LASTEXITCODE."
        }

        $remainingOutput = & python $countSubmittedScriptPath
        if ($LASTEXITCODE -ne 0) {
            throw "Submitted-batch count query exited with code $LASTEXITCODE."
        }

        $remainingSubmitted = [int]($remainingOutput | Select-Object -Last 1)
        Write-Host "Remaining submitted batches to poll: $remainingSubmitted"

        if ($remainingSubmitted -eq 0) {
            Write-Host "No submitted batches remain; exiting."
            break
        }

        Write-Host "Sleeping $SleepSeconds seconds..."
        Start-Sleep -Seconds $SleepSeconds
    }
}
finally {
    Pop-Location
    if ($countSubmittedScriptPath -and (Test-Path $countSubmittedScriptPath)) {
        Remove-Item $countSubmittedScriptPath -ErrorAction SilentlyContinue
    }
}
