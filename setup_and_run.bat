@echo off
REM ================================================================
REM  Paper 2 V2  --  Setup and Run Script
REM ================================================================
REM  Run this in Command Prompt (cmd.exe), NOT PowerShell.
REM  If you are in PowerShell, type:  cmd  first, then run this.
REM ================================================================

echo.
echo ============================================================
echo   Paper 2 V2:  MiniROCKET Cross-Subject sEMG Generalization
echo ============================================================
echo.

REM --- Step 1: Create virtual environment ---
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Could not create venv.  Make sure Python 3.8+ is installed.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists.
)

REM --- Step 2: Activate it ---
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM --- Step 3: Install dependencies ---
echo [3/4] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.  Check your internet connection.
    pause
    exit /b 1
)

REM --- Step 4: Menu ---
echo.
echo ============================================================
echo   Setup complete!  Choose what to run:
echo ============================================================
echo.
echo   1  Run FULL pipeline  (all days, all 15 experiments)
echo   2  Run only Day 2  (main LOSO experiments)
echo   3  Run only Day 2  with --resume  (continue from checkpoint)
echo   4  Run only Day 2b (ablation studies)
echo   5  Run only Day 3  (domain shift metrics)
echo   6  Run only Day 4  (kernel importance)
echo   7  Run only Day 5  (statistical tables)
echo   8  Run only Day 6  (all figures)
echo   9  Run specific DB + method  (you will be asked)
echo   0  Exit
echo.

set /p choice="Enter choice (0-9): "

if "%choice%"=="1" goto run_all
if "%choice%"=="2" goto run_day2
if "%choice%"=="3" goto run_day2_resume
if "%choice%"=="4" goto run_day2b
if "%choice%"=="5" goto run_day3
if "%choice%"=="6" goto run_day4
if "%choice%"=="7" goto run_day5
if "%choice%"=="8" goto run_day6
if "%choice%"=="9" goto run_specific
if "%choice%"=="0" goto end
goto invalid_choice

:run_all
echo.
echo Running FULL pipeline...
python run_full_pipeline.py --all
goto done

:run_day2
echo.
echo Running Day 2 (main LOSO, all experiments)...
python p01a_run_main_loso_benchmark.py --all
goto done

:run_day2_resume
echo.
echo Running Day 2 (resume from checkpoint)...
python p01a_run_main_loso_benchmark.py --all --resume
goto done

:run_day2b
echo.
echo Running Day 2b (ablation)...
python p02_run_window_and_sample_ablation.py
goto done

:run_day3
echo.
echo Running Day 3 (domain shift)...
python p04b_compute_domain_shift_metrics.py
goto done

:run_day4
echo.
echo Running Day 4 (kernel importance)...
python p05_analyze_kernel_importance.py
goto done

:run_day5
echo.
echo Running Day 5 (statistics)...
python p06_run_statistical_analysis.py
goto done

:run_day6
echo.
echo Running Day 6 (figures, skip t-SNE for speed)...
python p07_generate_all_figures.py --skip-tsne
goto done

:run_specific
echo.
echo Available databases: db7, db3, db2
set /p db="Database (db7/db3/db2): "
echo Available methods: raw, centering, coral, tca, sa
set /p method="Method: "
echo.
echo Running: p01a_run_main_loso_benchmark.py --db %db% --method %method% --resume
python p01a_run_main_loso_benchmark.py --db %db% --method %method% --resume
goto done

:invalid_choice
echo.
echo Invalid choice.  Please enter 0-9.
pause
exit /b 1

:done
echo.
echo ============================================================
echo   Finished!  Outputs are in:  outputs\
echo ============================================================
echo.
pause
:end