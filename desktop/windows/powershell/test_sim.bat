@echo off
setlocal enabledelayedexpansion

:: Default cores
set N_CORES=22

:: Parse arguments
:loop
if "%~1"=="" goto check_verbose
if "%~1"=="-nc" (
    if not "%~2"=="" (
        set N_CORES=%~2
        shift
    )
) else if "%~1"=="--verbose" (
    set VERBOSE=1
)
shift
goto loop

:check_verbose
:: If verbose mode is requested, turn echoing on
if %VERBOSE% equ 1 (
    @echo on
)

:config
:: Configuration
set SEED=42
set N_BINS=170
set N_DAYS=31
set N_SAMPLES=10
set PROBLEM=vrpp

set WTYPE=plastic
set AREA=riomaior
set DATA_DIST=emp
set ENV_FILE=vars.env
set IDX_PATH=graphs_170V_1N_plastic.json

:: Multiple policies as space-separated list
set POLICIES=policy_regular
:: To add more policies, use:
:: set POLICIES=policy_regular policy_look_ahead policy_last_minute gurobi_vrpp hexaly_vrpp am amgc transgcn

set REGULAR_LEVEL=3
set LAST_MINUTE_CF=50
set GUROBI_PARAM=0.84
set HEXALY_PARAM=0.84
set DECODE_TYPE=greedy
set LOOKAHEAD_CONFIGS=

set VEHICLES=1
set EDGE_THRESH=0.0
set EDGE_METHOD=knn
set VERTEX_METHOD=mmn
set DIST_METHOD=gmaps
set DM_PATH=data/wsr_simulator/distance_matrix/gmaps_distmat_plastic[riomaior].csv
set WASTE_PATH=daily_waste/riomaior170_emp_wsr31_N10_seed42.npz

set CHECKPOINTS=40

echo ========================================
echo Test Configuration
echo ========================================
echo Cores: %N_CORES%
echo Problem: %PROBLEM%
echo Area: %AREA%
echo Policies: %POLICIES%
echo Samples: %N_SAMPLES%
echo Days: %N_DAYS%
echo ========================================
echo.

if %VERBOSE% equ 0 (
    @echo on
)
python main.py test --policies %POLICIES% --data_distribution "%DATA_DIST%" --dt "%DECODE_TYPE%" --hp %HEXALY_PARAM% ^
    --n_samples %N_SAMPLES% --area "%AREA%" --focus_graph "%IDX_PATH%" --size %N_BINS% --seed %SEED% --dm "%DIST_METHOD%" ^
    --problem "%PROBLEM%" --n_vehicles %VEHICLES% --vm "%VERTEX_METHOD%" --env_file "%ENV_FILE%" ^
    --days %N_DAYS% --lvl %REGULAR_LEVEL% --cf %LAST_MINUTE_CF% --gp %GUROBI_PARAM% --lac %LOOKAHEAD_CONFIGS% ^
    --cc %N_CORES% --et %EDGE_THRESH% --em "%EDGE_METHOD%" --waste_type "%WTYPE%" --dm_filepath "%DM_PATH%" --cpd %CHECKPOINTS%
if %VERBOSE% equ 0 (
    @echo off
)

if !errorlevel! equ 0 (
    echo.
    echo ========================================
    echo Test completed successfully
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Test failed with error code !errorlevel!
    echo ========================================
    exit /b !errorlevel!
)

endlocal
