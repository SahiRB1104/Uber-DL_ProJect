# Uber Trip Analysis - PySpark + Parquet Pipeline (PowerShell)
# This script runs ETL and validates parquet outputs.

Write-Host ""
Write-Host "============================================================"
Write-Host "  UBER TRIP ANALYSIS - PYSPARK PARQUET PIPELINE" -ForegroundColor Cyan
Write-Host "============================================================"
Write-Host ""

# Set project variables
$ProjectRoot = "D:\PWA\Uber-Trip-Analysis"
$PythonBin = "python"
$UberRawInput = "$ProjectRoot\Data.csv"
$UberOutputBase = "$ProjectRoot\data"

# Set environment variables
Write-Host "[1/4] Setting environment variables..."
$env:UBER_PROJECT_ROOT = $ProjectRoot
$env:UBER_PYTHON_BIN = $PythonBin
$env:UBER_RAW_INPUT = $UberRawInput
$env:UBER_OUTPUT_BASE = $UberOutputBase
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

Write-Host "    ✓ UBER_PROJECT_ROOT = $ProjectRoot"
Write-Host "    ✓ UBER_RAW_INPUT = $UberRawInput"
Write-Host "    ✓ UBER_OUTPUT_BASE = $UberOutputBase"
Write-Host ""

# Create data output directory
Write-Host "[2/4] Creating output directory..."
if (-not (Test-Path $UberOutputBase)) {
    New-Item -ItemType Directory -Path $UberOutputBase -Force | Out-Null
    Write-Host "    ✓ Created $UberOutputBase"
} else {
    Write-Host "    ✓ $UberOutputBase already exists"
}
Write-Host ""

# Activate venv
Write-Host "[3/4] Activating venv..."
Push-Location $ProjectRoot
& "$ProjectRoot\venv\Scripts\Activate.ps1"
Write-Host "    ✓ venv activated"
Write-Host ""

# Run Spark ETL
Write-Host "[4/4] Running Spark ETL Pipeline..."
Write-Host "    Command: python spark_etl_pipeline.py --input `"$UberRawInput`" --output `"$UberOutputBase`""
& python spark_etl_pipeline.py --input "$UberRawInput" --output "$UberOutputBase"
if ($LASTEXITCODE -ne 0) {
    Write-Host "    ✗ Spark ETL failed!"
    Pop-Location
    exit 1
}
Write-Host "    ✓ Spark ETL completed successfully"
Write-Host ""

# Verify outputs
Write-Host "Verifying Parquet outputs..."
if (Test-Path "$UberOutputBase\curated_trips") {
    Write-Host "    ✓ curated_trips folder found"
} else {
    Write-Host "    ✗ curated_trips folder NOT found"
    Pop-Location
    exit 1
}
if (Test-Path "$UberOutputBase\daily_aggregates") {
    Write-Host "    ✓ daily_aggregates folder found"
} else {
    Write-Host "    ✗ daily_aggregates folder NOT found"
    Pop-Location
    exit 1
}
Write-Host ""

Pop-Location

Write-Host "============================================================"
Write-Host "  ✅ PIPELINE SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================"
Write-Host ""
Write-Host "Next Steps:"
Write-Host "  1. Run Dashboard:"
Write-Host "     cd `"$ProjectRoot`""
Write-Host "     & `"$ProjectRoot\venv\Scripts\Activate.ps1`""
Write-Host "     streamlit run dashboard.py"
Write-Host ""
Write-Host "  2. In sidebar, set parquet options:"
Write-Host "     Prefer curated parquet if available = ON"
Write-Host "     Curated parquet path = data/curated_trips"
Write-Host ""
Write-Host "  3. Re-run ETL anytime after CSV changes:"
Write-Host "     cd `"$ProjectRoot`""
Write-Host "     & `"$ProjectRoot\venv\Scripts\Activate.ps1`""
Write-Host "     python spark_etl_pipeline.py --input `"$UberRawInput`" --output `"$UberOutputBase`""
Write-Host ""
Write-Host "============================================================"
Write-Host ""
