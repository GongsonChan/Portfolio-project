$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

docker compose up -d | Out-Null

$maxWaitSec = 90
$elapsed = 0
while ($elapsed -lt $maxWaitSec) {
    docker exec smallfunds_db pg_isready -q 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 5
    $elapsed += 5
}

$python = Join-Path $root "..\.venv\Scripts\python.exe"
$script = Join-Path $root "scripts\daily_update_us.py"
$log = Join-Path $root "daily_update_us.log"

cmd /c "`"$python`" `"$script`" >> `"$log`" 2>&1"
