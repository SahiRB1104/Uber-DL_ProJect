@echo off
REM Uber Trip Analysis - PySpark + Parquet Pipeline
REM This script runs ETL and validates parquet outputs.

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  UBER TRIP ANALYSIS - PYSPARK PARQUET PIPELINE
echo ============================================================
echo.

REM Set project variables
set PROJECT_ROOT=D:\PWA\Uber-Trip-Analysis
set PYTHON_BIN=python
set UBER_RAW_INPUT=%PROJECT_ROOT%\Data.csv
set UBER_OUTPUT_BASE=%PROJECT_ROOT%\data

echo [1/4] Setting environment variables...
set UBER_PROJECT_ROOT=%PROJECT_ROOT%
set UBER_PYTHON_BIN=%PYTHON_BIN%
set UBER_RAW_INPUT=%UBER_RAW_INPUT%
set UBER_OUTPUT_BASE=%UBER_OUTPUT_BASE%
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

echo     ✓ UBER_PROJECT_ROOT = %PROJECT_ROOT%
echo     ✓ UBER_RAW_INPUT = %UBER_RAW_INPUT%
echo     ✓ UBER_OUTPUT_BASE = %UBER_OUTPUT_BASE%
echo.

REM Create data output directory if not exists
if not exist "%UBER_OUTPUT_BASE%" (
    echo [2/4] Creating output directory...
    mkdir "%UBER_OUTPUT_BASE%"
    echo     ✓ Created %UBER_OUTPUT_BASE%
) else (
    echo [2/4] Output directory already exists
    echo     ✓ %UBER_OUTPUT_BASE%
)
echo.

REM Activate venv
echo [3/4] Activating venv...
cd /d "%PROJECT_ROOT%"
call venv\Scripts\activate.bat
echo     ✓ venv activated
echo.

REM Run Spark ETL Pipeline
echo [4/4] Running Spark ETL Pipeline...
echo     Command: python spark_etl_pipeline.py --input "%UBER_RAW_INPUT%" --output "%UBER_OUTPUT_BASE%"
python spark_etl_pipeline.py --input "%UBER_RAW_INPUT%" --output "%UBER_OUTPUT_BASE%"
if errorlevel 1 (
    echo     ✗ Spark ETL failed!
    pause
    exit /b 1
)
echo     ✓ Spark ETL completed successfully
echo.

REM Verify outputs
echo Verifying Parquet outputs...
if exist "%UBER_OUTPUT_BASE%\curated_trips" (
    echo     ✓ curated_trips folder found
) else (
    echo     ✗ curated_trips folder NOT found
    pause
    exit /b 1
)
if exist "%UBER_OUTPUT_BASE%\daily_aggregates" (
    echo     ✓ daily_aggregates folder found
) else (
    echo     ✗ daily_aggregates folder NOT found
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  ✅ PIPELINE SETUP COMPLETE!
echo ============================================================
echo.
echo Next Steps:
echo   1. Run Dashboard:
echo      cd "%PROJECT_ROOT%"
echo      call venv\Scripts\activate.bat
echo      streamlit run dashboard.py
echo.
echo   2. In sidebar, enable preferred parquet mode:
echo      Prefer curated parquet if available = ON
echo      Curated parquet path = data\curated_trips
echo.
echo   3. Re-run ETL anytime after CSV changes:
echo      cd "%PROJECT_ROOT%"
echo      call venv\Scripts\activate.bat
echo      python spark_etl_pipeline.py --input "%UBER_RAW_INPUT%" --output "%UBER_OUTPUT_BASE%"
echo.
echo ============================================================
echo.
pause
