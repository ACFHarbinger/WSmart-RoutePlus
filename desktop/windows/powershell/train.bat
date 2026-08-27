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
set EDGE_T=1.0
set EDGE_M=knn
set DIST_M=gmaps
set VERTEX_M=mmn

set W_LEN=1.75
set W_OVER=0.0
set W_WASTE=1.0

set EMBED_DIM=128
set HIDDEN_DIM=512
set N_ENC_L=3
set N_ENC_SL=1
set N_PRED_L=2
set N_DEC_L=2

set N_HEADS=8
set NORM=instance
set ACTI_F=gelu
set DROPOUT=0.1
set AGG=sum
set AGG_G=avg

set OPTIM=rmsprop
set LR_MODEL=0.0001
set LR_CV=0.0001
set LR_SCHEDULER=lambda
set LR_DECAY=1.0

set B_SIZE=256
set N_DATA=128000
set N_VAL_DATA=0
set VAL_B_SIZE=0

set BL=
set MAX_NORM=1.0
set EXP_BETA=0.8
set BL_ALPHA=0.05
set ACC_STEPS=1

set SIZE=170
set AREA=riomaior
set WTYPE=plastic
set F_SIZE=128000
set VAL_F_SIZE=0
set DM_METHOD=gmaps
set F_GRAPH=graphs_170V_1N_plastic.json
set DM_PATH=data/wsr_simulator/distance_matrix/gmaps_distmat_plastic[riomaior].csv

:: Note: N_BINS, N_DAYS, N_SAMPLES not defined in original, using defaults
set N_BINS=170
set N_DAYS=31
set N_SAMPLES=10
set WASTE_PATH=daily_waste/riomaior170_gamma1_wsr31_N10_seed42.npz

set SEED=42
set START=0
set EPOCHS=31
set /A TOTAL_EPOCHS=%START% + %EPOCHS%
set PROBLEM=vrpp
set DATASET_NAME=time%TOTAL_EPOCHS%
set VAL_DATASET_NAME=time%TOTAL_EPOCHS%_val

:: Data distributions - using single distribution for simplicity
set DATA_DISTS=gamma1
:: For multiple distributions, use: set DATA_DISTS=gamma1 gamma2 emp

:: Datasets array simulation
set DATASET_1=data/datasets/%PROBLEM%/%PROBLEM%%SIZE%_gamma1_%DATASET_NAME%_seed%SEED%.pkl

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

echo Starting training script...
echo Problem: %PROBLEM%
echo Graph size: %SIZE%
echo Area: %AREA%
echo Epochs: %EPOCHS%
echo.

:: Loop through data distributions
for %%D in (%DATA_DISTS%) do (
    echo Processing data distribution: %%D

    :: Update dataset path for current distribution
    set CURRENT_DATASET=data/datasets/%PROBLEM%/%PROBLEM%%SIZE%_%%D_%DATASET_NAME%_seed%SEED%.pkl

    :: Train AM model
    if %TRAIN_AM% equ 0 (
        echo ===== Training AM model =====
        if %VERBOSE% equ 0 (
            @echo on
        )
        python main.py train --problem "%PROBLEM%" --model am --encoder gat --epoch_size %N_DATA% ^
        --data_distribution "%%D" --graph_size %SIZE% --n_epochs %EPOCHS% --seed %SEED% ^
        --train_time --vertex_method "%VERTEX_M%" --epoch_start %START% --max_grad_norm %MAX_NORM% ^
        --val_size %N_VAL_DATA% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% ^
        --embed_dim %EMBED_DIM% --activation "%ACTI_F%" --accumulation_steps %ACC_STEPS% ^
        --focus_graph "%F_GRAPH%" --normalization "%NORM%" --train_dataset "!CURRENT_DATASET!" ^
        --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --n_heads %N_HEADS% --dropout %DROPOUT% ^
        --waste_type "%WTYPE%" --focus_size %F_SIZE% --n_encode_layers %N_ENC_L% --lr_model %LR_MODEL% ^
        --eval_focus_size %VAL_F_SIZE% --distance_method "%DIST_M%" --exp_beta %EXP_BETA% ^
        --edge_threshold %EDGE_T% --edge_method "%EDGE_M%" --eval_batch_size %VAL_B_SIZE% ^
        --temporal_horizon %HORIZON_0% --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% ^
        --batch_size %B_SIZE% --lr_critic_value %LR_CV% --bl_alpha %BL_ALPHA% --area "%AREA%" ^
        --aggregation_graph "%AGG_G%" --distance_method "%DM_METHOD%" --dm_filepath "%DM_PATH%" ^
        --wandb_mode "%WB_MODE%" --distance_method "%DM_METHOD%"
        if %VERBOSE% equ 0 (
            @echo off
        )
        if !errorlevel! neq 0 (
            echo ERROR: AM training failed for distribution %%D
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
        python main.py train --problem "%PROBLEM%" --model am --encoder gac --epoch_size %N_DATA% ^
        --data_distribution "%%D" --graph_size %SIZE% --n_epochs %EPOCHS% --seed %SEED% ^
        --train_time --vertex_method "%VERTEX_M%" --epoch_start %START% --max_grad_norm %MAX_NORM% ^
        --val_size %N_VAL_DATA% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% ^
        --embed_dim %EMBED_DIM% --activation "%ACTI_F%" --accumulation_steps %ACC_STEPS% ^
        --focus_graph "%F_GRAPH%" --normalization "%NORM%" --train_dataset "!CURRENT_DATASET!" ^
        --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --n_heads %N_HEADS% --dropout %DROPOUT% ^
        --waste_type "%WTYPE%" --focus_size %F_SIZE% --n_encode_layers %N_ENC_L% --lr_model %LR_MODEL% ^
        --eval_focus_size %VAL_F_SIZE% --distance_method "%DIST_M%" --exp_beta %EXP_BETA% ^
        --edge_threshold %EDGE_T% --edge_method "%EDGE_M%" --eval_batch_size %VAL_B_SIZE% ^
        --temporal_horizon %HORIZON_1% --lr_scheduler "%LR_SCHEDULER%" --lr_decay %LR_DECAY% ^
        --batch_size %B_SIZE% --lr_critic_value %LR_CV% --bl_alpha %BL_ALPHA% --area "%AREA%" ^
        --aggregation_graph "%AGG_G%" --distance_method "%DM_METHOD%" --dm_filepath "%DM_PATH%" ^
        --wandb_mode "%WB_MODE%" --distance_method "%DM_METHOD%"
        if %VERBOSE% equ 0 (
            @echo off
        )
        if !errorlevel! neq 0 (
            echo ERROR: AMGC training failed for distribution %%D
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
        python main.py train --problem "%PROBLEM%" --model am --encoder tgc --epoch_size %N_DATA% ^
        --data_distribution "%%D" --graph_size %SIZE% --n_epochs %EPOCHS% ^
        --train_time --vertex_method "%VERTEX_M%" --epoch_start %START% --max_grad_norm %MAX_NORM% ^
        --val_size %N_VAL_DATA% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% ^
        --embed_dim %EMBED_DIM% --activation "%ACTI_F%" --accumulation_steps %ACC_STEPS% ^
        --focus_graph "%F_GRAPH%" --normalization "%NORM%" --train_dataset "!CURRENT_DATASET!" ^
        --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --n_heads %N_HEADS% --dropout %DROPOUT% ^
        --waste_type "%WTYPE%" --focus_size %F_SIZE% --n_encode_layers %N_ENC_L% --lr_model %LR_MODEL% ^
        --eval_focus_size %VAL_F_SIZE% --distance_method "%DIST_M%" --exp_beta %EXP_BETA% --area "%AREA%" ^
        --edge_threshold %EDGE_T% --edge_method "%EDGE_M%" --eval_batch_size %VAL_B_SIZE% --seed %SEED% ^
        --temporal_horizon %HORIZON_2% --lr_scheduler "%LR_SCHEDULER%" --n_encode_sublayers %N_ENC_SL% ^
        --batch_size %B_SIZE% --lr_critic_value %LR_CV% --bl_alpha %BL_ALPHA% --lr_decay %LR_DECAY% ^
        --aggregation_graph "%AGG_G%" --distance_method "%DM_METHOD%" --dm_filepath "%DM_PATH%" ^
        --wandb_mode "%WB_MODE%" --distance_method "%DM_METHOD%"
        if %VERBOSE% equ 0 (
            @echo off
        )
        if !errorlevel! neq 0 (
            echo ERROR: TRANSGCN training failed for distribution %%D
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
        python main.py train --problem "%PROBLEM%" --model ddam --encoder gat --epoch_size %N_DATA% ^
        --data_distribution "%%D" --graph_size %SIZE% --n_epochs %EPOCHS% --seed %SEED% ^
        --train_time --vertex_method "%VERTEX_M%" --epoch_start %START% --max_grad_norm %MAX_NORM% ^
        --val_size %N_VAL_DATA% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% ^
        --embed_dim %EMBED_DIM% --activation "%ACTI_F%" --accumulation_steps %ACC_STEPS% ^
        --focus_graph "%F_GRAPH%" --normalization "%NORM%" --train_dataset "!CURRENT_DATASET!" --area "%AREA%" ^
        --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --n_heads %N_HEADS% --dropout %DROPOUT% ^
        --waste_type "%WTYPE%" --focus_size %F_SIZE% --n_encode_layers %N_ENC_L% --lr_model %LR_MODEL% ^
        --eval_focus_size %VAL_F_SIZE% --distance_method "%DIST_M%" --exp_beta %EXP_BETA% ^
        --edge_threshold %EDGE_T% --edge_method "%EDGE_M%" --eval_batch_size %VAL_B_SIZE% ^
        --temporal_horizon %HORIZON_3% --lr_scheduler "%LR_SCHEDULER%" --n_decode_layers %N_DEC_L% ^
        --batch_size %B_SIZE% --lr_critic_value %LR_CV% --bl_alpha %BL_ALPHA% --lr_decay %LR_DECAY% ^
        --aggregation_graph "%AGG_G%" --distance_method "%DM_METHOD%" --dm_filepath "%DM_PATH%" ^
        --wandb_mode "%WB_MODE%" --distance_method "%DM_METHOD%"
        if %VERBOSE% equ 0 (
            @echo off
        )
        if !errorlevel! neq 0 (
            echo ERROR: DDAM training failed for distribution %%D
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
        python main.py train --problem "%PROBLEM%" --model tam --encoder gat --epoch_size %N_DATA% ^
        --data_distribution "%%D" --graph_size %SIZE% --n_epochs %EPOCHS% --seed %SEED% ^
        --train_time --vertex_method "%VERTEX_M%" --epoch_start %START% --max_grad_norm %MAX_NORM% ^
        --val_size %N_VAL_DATA% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% ^
        --embed_dim %EMBED_DIM% --activation "%ACTI_F%" --accumulation_steps %ACC_STEPS% ^
        --focus_graph "%F_GRAPH%" --normalization "%NORM%" --train_dataset "!CURRENT_DATASET!" --area "%AREA%" ^
        --optimizer "%OPTIM%" --hidden_dim %HIDDEN_DIM% --n_heads %N_HEADS% --dropout %DROPOUT% ^
        --waste_type "%WTYPE%" --focus_size %F_SIZE% --n_encode_layers %N_ENC_L% --lr_model %LR_MODEL% ^
        --eval_focus_size %VAL_F_SIZE% --distance_method "%DIST_M%" --exp_beta %EXP_BETA% ^
        --edge_threshold %EDGE_T% --edge_method "%EDGE_M%" --eval_batch_size %VAL_B_SIZE% ^
        --temporal_horizon %HORIZON_4% --lr_scheduler "%LR_SCHEDULER%" --n_predict_layers %N_PRED_L% ^
        --batch_size %B_SIZE% --lr_critic_value %LR_CV% --bl_alpha %BL_ALPHA% --lr_decay %LR_DECAY% ^
        --aggregation_graph "%AGG_G%" --distance_method "%DM_METHOD%" --dm_filepath "%DM_PATH%" ^
        --wandb_mode "%WB_MODE%" --distance_method "%DM_METHOD%"
        if %VERBOSE% equ 0 (
            @echo off
        )
        if !errorlevel! neq 0 (
            echo ERROR: TAM training failed for distribution %%D
            exit /b 1
        )
    ) else (
        echo Skipping TAM training (TRAIN_TAM=%TRAIN_TAM%)
    )
)

echo.
echo ===== Training completed for all distributions =====
endlocal
