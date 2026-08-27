@echo off
setlocal enabledelayedexpansion

:: Default verbose mode
set VERBOSE=0

:: Parse arguments for verbose flag first
:parse_args
if "%~1"=="" goto check_verbose
if "%~1"=="--verbose" (
    set VERBOSE=1
)
shift
goto parse_args

:check_verbose
:: If verbose mode is requested, turn echoing on
if %VERBOSE% equ 1 (
    @echo on
)

:: Simple configuration - no arrays
set SEED=42
set START=0
set N_EPOCHS=31
set PROBLEM=vrpp
set AREA=riomaior
set VERTEX_METHOD=mmn
set WTYPE=plastic

:: Generation flags
set GENERATE_DATASET=0
set GENERATE_VAL_DATASET=1
set GENERATE_TEST_DATASET=1

echo Starting data generation...

:: Main dataset
if %GENERATE_DATASET% equ 0 (
    echo Generating main dataset...
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py gen_data --name "time" --problem "%PROBLEM%" -f ^
    --waste_type "%WTYPE%" --graph_sizes 20 50 100 170 --dataset_size 128000 ^
    --focus_graph "graphs_20V_1N_plastic.json" "graphs_50V_1N_plastic.json" "graphs_100V_1N_plastic.json" "graphs_170V_1N_plastic.json" ^
    --focus_size 128000 --data_dir "datasets" --area "%AREA%" --vertex_method "%VERTEX_METHOD%" ^
    --epoch_start %START% --seed %SEED% --n_epochs %N_EPOCHS% --data_distribution "gamma1" --dataset_type "train_time"
    if %VERBOSE% equ 0 (
        @echo off
    )
)

:: Validation dataset
if %GENERATE_VAL_DATASET% equ 0 (
    echo Generating validation dataset...
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py gen_data --name "time_val" --problem "%PROBLEM%" -f ^
    --waste_type "%WTYPE%" --graph_sizes 20 50 100 170 --dataset_size 1280 ^
    --area "%AREA%" --vertex_method "%VERTEX_METHOD%" --epoch_start %START% --seed %SEED% ^
    --n_epochs %N_EPOCHS% --data_distribution "gamma1" --dataset_type "train_time" ^
    --focus_graph "graphs_20V_1N_plastic.json" "graphs_50V_1N_plastic.json" "graphs_100V_1N_plastic.json" "graphs_170V_1N_plastic.json" ^
    --focus_size 1280 --data_dir "datasets"
    if %VERBOSE% equ 0 (
        @echo off
    )
)

:: Test dataset
if %GENERATE_TEST_DATASET% equ 0 (
    echo Generating test dataset...
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py gen_data --name "wsr" --problem "%PROBLEM%" -f ^
    --area "%AREA%" --vertex_method "%VERTEX_METHOD%" --epoch_start %START% --seed %SEED% ^
    --n_epochs %N_EPOCHS% --data_distribution "gamma1" --dataset_type "test_simulator" ^
    --focus_graph "graphs_20V_1N_plastic.json" "graphs_50V_1N_plastic.json" "graphs_100V_1N_plastic.json" "graphs_170V_1N_plastic.json" ^
    --focus_size 1 --data_dir "daily_waste" --waste_type "%WTYPE%" --graph_sizes 20 50 100 170 --dataset_size 10
    if %VERBOSE% equ 0 (
        @echo off
    )
)

echo Done!
endlocal
