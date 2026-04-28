# scripts/run_clean_tests.ps1
# Run the clean model tests and invoke the conclusiveness checker.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = Split-Path $PSScriptRoot -Parent

Write-Host "=== Installing / verifying dependencies ==="
python -m pip install torch numpy pytest --quiet

Write-Host ""
Write-Host "=== Running tests ==="
Set-Location $repo
python -m pytest -v tests\test_model_clean.py
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=== Conclusiveness check ==="
& "$PSScriptRoot\check_conclusiveness.ps1" -TestExitCode $exitCode

exit $exitCode