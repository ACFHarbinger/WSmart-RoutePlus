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

:: Configuration variables
set START=0
set EPOCHS=7

set EDGE_T=0.3
set EDGE_M=knn
set VERTEX_M=mmn
set DATA_DIST=gamma1

set W_LEN=1.0
set W_OVER=1.0
set W_WASTE=1.0

set EMBED_DIM=128
set HIDDEN_DIM=512
set N_ENC_L=3
set N_ENC_SL=1
set N_PRED_L=2
set N_DEC_L=2

set N_HEADS=8
set NORM=instance
set ACTIVATION=gelu
set DROPOUT=0.1
set AGG=mean
set AGG_G=mean

set OPTIM=rmsprop
set LR_MODEL=0.0001
set LR_CV=0.0001
set LR_SCHEDULER=lambda
set LR_DECAY=1.0

set B_SIZE=256
set N_DATA=128000
set N_VAL_DATA=1280
set VAL_B_SIZE=256

set BL=exponential
set MAX_NORM=1.0
set EXP_BETA=0.8
set BL_ALPHA=0.05
set ACC_STEPS=1

set ETA=5
set N_POP=20
set FEVALS=25
set METRIC=both
set HPO_METHOD=dehb
set RANGE=0.0 2.0
set MAX_TRES=40
set H_EPOCHS=3
set MUTPB=0.3
set CXPB=0.5

set SIZE=20
set AREA=Rio Maior
set WTYPE=plastic
set F_SIZE=1
set VAL_F_SIZE=1280
set FOCUS_GRAPH=graphs_20V_1N_plastic.json

set SEED=42
set PROBLEM=wcvrp
set DATASET_NAME=real
set VAL_DATASET=real_val
set DATA_DIST=gamma1
:: VAL_DATASET="data/datasets/%PROBLEM%/%PROBLEM%%SIZE%_%DATA_DIST%_%VAL_DATASET%_seed%SEED%.pkl"
set DATASET=data/datasets/%PROBLEM%/%PROBLEM%%SIZE%_%DATA_DIST%_%DATASET_NAME%_seed%SEED%.pkl

:: Training flags
set TRAIN_AM=0
set TRAIN_AMGC=1
set TRAIN_TRANSGCN=1
set TRAIN_DDAM=1
set TRAIN_TAM=1

:: Horizon array simulation
set HORIZON_0=0
set HORIZON_1=0
set HORIZON_2=0
set HORIZON_3=0
set HORIZON_4=3

set WB_MODE=disabled

echo Starting hyperparameter optimization...
echo Problem: %PROBLEM%
echo Graph size: %SIZE%
echo Area: %AREA%
echo.

:: Train AM model
if %TRAIN_AM% equ 0 (
    echo ===== Training AM model =====
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py hp_optim --problem "%PROBLEM%" --model am --encoder gat --hop_epochs %H_EPOCHS% --waste_type "%WTYPE%" ^
    --data_distribution "%DATA_DIST%" --n_epochs %EPOCHS% --vertex_method "%VERTEX_M%" --eta %ETA% --mutpb %MUTPB% ^
    --batch_size %B_SIZE% --epoch_size %N_DATA% --val_size %N_VAL_DATA% --temporal_horizon %HORIZON_0% ^
    --train_time --train_dataset "%DATASET%" --normalization "%NORM%" --embed_dim %EMBED_DIM% --cxpb %CXPB% ^
    --area "%AREA%" --activation "%ACTIVATION%" --n_encode_layers %N_ENC_L% --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% ^
    --graph_size %SIZE% --no_tensorboard --range %RANGE% --fevals %FEVALS% --metric "%METRIC%" --n_pop %N_POP% ^
    --focus_size %F_SIZE% --eval_focus_size %VAL_F_SIZE% --focus_graph "%FOCUS_GRAPH%" --exp_beta %EXP_BETA% ^
    --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% --lr_model %LR_MODEL% --lr_critic_value %LR_CV% --dropout %DROPOUT% ^
    --eval_batch_size %VAL_B_SIZE% --aggregation_graph "%AGG_G%" --max_grad_norm %MAX_NORM% --accumulation_steps %ACC_STEPS% ^
    --seed %SEED% --n_heads %N_HEADS% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% --epoch_start %START% ^
    --wandb_mode "%WB_MODE%"
    if %VERBOSE% equ 0 (
        @echo off
    )
    if !errorlevel! neq 0 (
        echo ERROR: AM training failed
        exit /b 1
    )
) else (
    echo Skipping AM training (TRAIN_AM=%TRAIN_AM%)
)

:: Train AMGC model
if %TRAIN_AMGC% equ 0 (
    echo.
    echo ===== Training AMGC model =====
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py hp_optim --problem "%PROBLEM%" --model am --encoder gac --hop_epochs %H_EPOCHS% --waste_type "%WTYPE%" ^
    --data_distribution "%DATA_DIST%" --n_epochs %EPOCHS% --vertex_method "%VERTEX_M%" --mutpb %MUTPB% --cxpb %CXPB% ^
    --batch_size %B_SIZE% --epoch_size %N_DATA% --val_size %N_VAL_DATA% --n_pop %N_POP% --edge_method "%EDGE_M%" --eta %ETA% ^
    --train_dataset "%DATASET%" --normalization "%NORM%" --embed_dim %EMBED_DIM% --edge_threshold %EDGE_T% --area "%AREA%" ^
    --train_time --activation "%ACTIVATION%" --n_encode_layers %N_ENC_L% --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% ^
    --no_tensorboard --graph_size %SIZE% --range %RANGE% --fevals %FEVALS% --metric "%METRIC%" --aggregation "%AGG%" ^
    --focus_size %F_SIZE% --eval_focus_size %VAL_F_SIZE% --focus_graph "%FOCUS_GRAPH%" --temporal_horizon %HORIZON_1% ^
    --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% --lr_model %LR_MODEL% --lr_critic_value %LR_CV% --dropout %DROPOUT% ^
    --eval_batch_size %VAL_B_SIZE% --aggregation_graph "%AGG_G%" --max_grad_norm %MAX_NORM% --accumulation_steps %ACC_STEPS% ^
    --seed %SEED% --n_heads %N_HEADS% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% --epoch_start %START% ^
    --wandb_mode "%WB_MODE%"
    if %VERBOSE% equ 0 (
        @echo off
    )
    if !errorlevel! neq 0 (
        echo ERROR: AMGC training failed
        exit /b 1
    )
) else (
    echo Skipping AMGC training (TRAIN_AMGC=%TRAIN_AMGC%)
)

:: Train TRANSGCN model
if %TRAIN_TRANSGCN% equ 0 (
    echo.
    echo ===== Training TRANSGCN model =====
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py hp_optim --problem "%PROBLEM%" --model am --encoder tgc --hop_epochs %H_EPOCHS% --waste_type "%WTYPE%" ^
    --data_distribution "%DATA_DIST%" --n_epochs %EPOCHS% --vertex_method "%VERTEX_M%" --eta %ETA% --mutpb %MUTPB% ^
    --batch_size %B_SIZE% --epoch_size %N_DATA% --val_size %N_VAL_DATA% --n_pop %N_POP% --area "%AREA%" ^
    --train_time --train_dataset "%DATASET%" --normalization "%NORM%" --embed_dim %EMBED_DIM% --aggregation "%AGG%" ^
    --activation "%ACTIVATION%" --n_encode_layers %N_ENC_L% --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% ^
    --graph_size %SIZE% --no_tensorboard --range %RANGE% --fevals %FEVALS% --metric "%METRIC%" --cxpb %CXPB% ^
    --focus_size %F_SIZE% --eval_focus_size %VAL_F_SIZE% --focus_graph "%FOCUS_GRAPH%" --temporal_horizon %HORIZON_2% ^
    --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% --lr_model %LR_MODEL% --lr_critic_value %LR_CV% ^
    --n_encode_sublayers %N_ENC_SL% --eval_batch_size %VAL_B_SIZE% --edge_method "%EDGE_M%" --edge_threshold %EDGE_T% ^
    --aggregation_graph "%AGG_G%" --max_grad_norm %MAX_NORM% --accumulation_steps %ACC_STEPS% --dropout %DROPOUT% ^
    --seed %SEED% --n_heads %N_HEADS% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% --epoch_start %START% ^
    --wandb_mode "%WB_MODE%"
    if %VERBOSE% equ 0 (
        @echo off
    )
    if !errorlevel! neq 0 (
        echo ERROR: TRANSGCN training failed
        exit /b 1
    )
) else (
    echo Skipping TRANSGCN training (TRAIN_TRANSGCN=%TRAIN_TRANSGCN%)
)

:: Train DDAM model
if %TRAIN_DDAM% equ 0 (
    echo.
    echo ===== Training DDAM model =====
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py hp_optim --problem "%PROBLEM%" --model ddam --encoder gat --hop_epochs %H_EPOCHS% --waste_type "%WTYPE%" ^
    --data_distribution "%DATA_DIST%" --n_epochs %EPOCHS% --vertex_method "%VERTEX_M%" --eta %ETA% --mutpb %MUTPB% ^
    --batch_size %B_SIZE% --epoch_size %N_DATA% --val_size %N_VAL_DATA% --n_pop %N_POP% --temporal_horizon %HORIZON_3% ^
    --train_time --train_dataset "%DATASET%" --normalization "%NORM%" --embed_dim %EMBED_DIM% --cxpb %CXPB% ^
    --activation "%ACTIVATION%" --n_encode_layers %N_ENC_L% --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --exp_beta %EXP_BETA% ^
    --graph_size %SIZE% --no_tensorboard --range %RANGE% --fevals %FEVALS% --metric "%METRIC%" --area "%AREA%" ^
    --focus_size %F_SIZE% --eval_focus_size %VAL_F_SIZE% --focus_graph "%FOCUS_GRAPH%" --n_decode_layers %N_DEC_L% ^
    --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% --lr_model %LR_MODEL% --lr_critic_value %LR_CV% --dropout %DROPOUT% ^
    --eval_batch_size %VAL_B_SIZE% --aggregation_graph "%AGG_G%" --max_grad_norm %MAX_NORM% --accumulation_steps %ACC_STEPS% ^
    --seed %SEED% --n_heads %N_HEADS% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% --epoch_start %START% ^
    --wandb_mode "%WB_MODE%"
    if %VERBOSE% equ 0 (
        @echo off
    )
    if !errorlevel! neq 0 (
        echo ERROR: DDAM training failed
        exit /b 1
    )
) else (
    echo Skipping DDAM training (TRAIN_DDAM=%TRAIN_DDAM%)
)

:: Train TAM model
if %TRAIN_TAM% equ 0 (
    echo.
    echo ===== Training TAM model =====
    if %VERBOSE% equ 0 (
        @echo on
    )
    python main.py hp_optim --problem "%PROBLEM%" --model tam --encoder gat --hop_epochs %H_EPOCHS% --waste_type "%WTYPE%" ^
    --metric "%METRIC%" --data_distribution "%DATA_DIST%" --n_epochs %EPOCHS% --vertex_method "%VERTEX_M%" --mutpb %MUTPB% ^
    --batch_size %B_SIZE% --epoch_size %N_DATA% --val_size %N_VAL_DATA% --n_pop %N_POP% --temporal_horizon %HORIZON_4% ^
    --area "%AREA%" --train_time --train_dataset "%DATASET%" --normalization "%NORM%" --embed_dim %EMBED_DIM% --cxpb %CXPB% ^
    --activation "%ACTIVATION%" --n_encode_layers %N_ENC_L% --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --exp_beta %EXP_BETA% ^
    --graph_size %SIZE% --no_tensorboard --range %RANGE% --fevals %FEVALS% --temporal_horizon %HORIZON_4% ^
    --focus_size %F_SIZE% --eval_focus_size %VAL_F_SIZE% --focus_graph "%FOCUS_GRAPH%" --n_predict_layers %N_PRED_L% --eta %ETA% ^
    --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% --lr_model %LR_MODEL% --lr_critic_value %LR_CV% --dropout %DROPOUT% ^
    --eval_batch_size %VAL_B_SIZE% --aggregation_graph "%AGG_G%" --max_grad_norm %MAX_NORM% --accumulation_steps %ACC_STEPS% ^
    --seed %SEED% --n_heads %N_HEADS% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% --epoch_start %START% ^
    --wandb_mode "%WB_MODE%"
    if %VERBOSE% equ 0 (
        @echo off
    )
    if !errorlevel! neq 0 (
        echo ERROR: TAM training failed
        exit /b 1
    )
) else (
    echo Skipping TAM training (TRAIN_TAM=%TRAIN_TAM%)
)

echo.
echo ===== Hyperparameter optimization completed =====
endlocal
