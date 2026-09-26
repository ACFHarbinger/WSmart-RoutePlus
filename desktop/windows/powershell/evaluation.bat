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

REM ==============================================================================
REM DEFAULT CONFIGURATION
REM Set default values for all evaluation parameters
REM ==============================================================================

REM Core Evaluation Settings
set "DECODE_TYPE=greedy"
set "DECODE_STRATEGY=greedy"
set "SOFTMAX_TEMPERATURE=1.0"
set "VAL_SIZE=12800"
set "OFFSET=0"
set "EVAL_BATCH_SIZE=256"
set "MAX_CALC_BATCH_SIZE=12800"
set "RESULTS_DIR=results_eval"
set "MODEL_PATH=checkpoints/best_model.pt"

REM Data/Dataset Configuration
set "DATASETS=data/example_vrp_50.pkl"
set "WIDTH=0"
set "GRAPH_SIZE=50"
set "AREA=riomaior"
set "WASTE_TYPE=plastic"

REM Graph Configuration
set "FOCUS_GRAPH="
set "FOCUS_SIZE=0"
set "EDGE_THRESHOLD=0"
set "EDGE_METHOD=knn"
set "DISTANCE_METHOD=ogd"
set "VERTEX_METHOD=mmn"

REM Flags (Default to false/unset)
set "OVERWRITE_FLAG="
set "NO_CUDA_FLAG="
set "NO_PROGRESS_BAR_FLAG="
set "COMPRESS_MASK_FLAG="
set "MULTIPROCESSING_FLAG="

REM Optional Output
set "OUTPUT_FILE="

REM ==============================================================================
REM CONSTRUCT COMMAND
REM Build the final Python evaluation command
REM ==============================================================================

REM Build the Python command
set "PYTHON_CMD=python main.py eval --model "!MODEL_PATH!" --datasets !DATASETS! !OVERWRITE_FLAG! !OUTPUT_ARG! --val_size "!VAL_SIZE!" --offset "!OFFSET!" --eval_batch_size "!EVAL_BATCH_SIZE!" --strategy "!DECODE_TYPE!" --width !WIDTH! --decode_strategy "!DECODE_STRATEGY!" --softmax_temperature "!SOFTMAX_TEMPERATURE!" !NO_CUDA_FLAG! !NO_PROGRESS_BAR_FLAG! !COMPRESS_MASK_FLAG! --max_calc_batch_size "!MAX_CALC_BATCH_SIZE!" --results_dir "!RESULTS_DIR!" !MULTIPROCESSING_FLAG! --graph_size "!GRAPH_SIZE!" --area "!AREA!" --waste_type "!WASTE_TYPE!" --focus_size "!FOCUS_SIZE!" --edge_threshold "!EDGE_THRESHOLD!" --edge_method "!EDGE_METHOD!" --distance_method "!DISTANCE_METHOD!" --vertex_method "!VERTEX_METHOD!""

REM Add focus_graph only if it is explicitly set
if not "!FOCUS_GRAPH!"=="" (
    set "PYTHON_CMD=!PYTHON_CMD! --focus_graph "!FOCUS_GRAPH!""
)

REM ==============================================================================
REM EXECUTION
REM ==============================================================================

if "!VERBOSE!"=="true" (
    echo ==========================================
    echo Starting Algorithm Evaluation
    echo Model: !MODEL_PATH!
    echo Datasets: !DATASETS!
    echo Decode Strategy: !DECODE_STRATEGY! ^(Width: !WIDTH!^)
    echo Output File: !OUTPUT_FILE!
    echo ==========================================
    echo.
)

REM Execute the command with appropriate output redirection
if "!VERBOSE!"=="true" (
    !PYTHON_CMD!
) else (
    !PYTHON_CMD! >nul 2>&1
)

REM Check exit status
if !errorlevel! equ 0 (
    if "!VERBOSE!"=="true" (
        echo.
        echo ==========================================
        echo Evaluation completed successfully.
        echo ==========================================
    ) else (
        echo Evaluation completed successfully.
    )
) else (
    if "!VERBOSE!"=="true" (
        echo.
        echo ==========================================
        echo ERROR: Evaluation failed.
        echo ==========================================
    ) else (
        echo ERROR: Evaluation failed.
    )
    exit /b 1
)

endlocal
