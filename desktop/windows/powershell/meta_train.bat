@echo off
setlocal enabledelayedexpansion

:: Default to quiet mode
set VERBOSE=false

:: Handle --verbose argument
for %%a in (%*) do (
    if "%%a"=="--verbose" set VERBOSE=true
)

:: Model Hyperparameters
set EDGE_T=1.0
set EDGE_M=knn
set DIST_M=gmaps
set VERTEX_M=mmn
set W_LEN=1.0
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
set N_DATA=1280
set N_VAL_DATA=0
set VAL_B_SIZE=0
set MAX_NORM=1.0
set EXP_BETA=0.8
set BL_ALPHA=0.05
set ACC_STEPS=1

:: Meta/HRL Parameters
set GAT_HIDDEN=128
set LSTM_HIDDEN=64
set GATE_THRESH=0.7
set META_METHOD=hrl
set META_HISTORY=10
set META_LR=0.0003
set META_STEP=10
set HRL_EPOCHS=4
set HRL_CLIP_EPS=0.2

:: Data and Environment
set SIZE=20
set AREA=riomaior
set WTYPE=plastic
set F_SIZE=1280
set VAL_F_SIZE=0
set DM_METHOD=gmaps
set F_GRAPH=graphs_%SIZE%V_1N_%WTYPE%.json
set DM_PATH=data/wsr_simulator/distance_matrix/gmaps_distmat_%WTYPE%[%AREA%].csv

set SEED=42
set START=0
set EPOCHS=31
set /a TOTAL_EPOCHS=%START% + %EPOCHS%
set PROBLEM=cvrpp
set DATA_PROBLEM=vrpp
set DATASET_NAME=time%TOTAL_EPOCHS%

:: Simulating Bash Array for Distributions
set DATA_DISTS_0=gamma1
set DIST_COUNT=0

:: Control Flags
set TRAIN_AM=0
set TRAIN_AMGC=1
set TRAIN_TRANSGCN=1
set TRAIN_DDAM=1
set TRAIN_TAM=1

:: Horizon Values
set H_0=0
set H_1=0
set H_2=0
set H_3=0
set H_4=3

set WB_MODE=disabled

echo Starting training script...
echo Problem: %PROBLEM%
echo Graph size: %SIZE%
echo Area: %AREA%
echo Epochs: %EPOCHS%
echo.

:: Loop through distributions
for /L %%i in (0,1,%DIST_COUNT%) do (
    set dist=!DATA_DISTS_%%i!
    set current_dataset=data/datasets/%DATA_PROBLEM%/%DATA_PROBLEM%%SIZE%_!dist!_%DATASET_NAME%_seed%SEED%.pkl

    echo Processing data distribution: !dist!

    if "%TRAIN_AM%"=="0" (
        echo ===== Training AM model =====
        set CMD=python main.py mrl_train --problem %PROBLEM% --model am --encoder gat --epoch_size %N_DATA% ^
        --data_distribution !dist! --graph_size %SIZE% --n_epochs %EPOCHS% --seed %SEED% ^
        --train_time --vertex_method %VERTEX_M% --epoch_start %START% --max_grad_norm %MAX_NORM% ^
        --val_size %N_VAL_DATA% --w_length %W_LEN% --w_waste %W_WASTE% --w_overflows %W_OVER% ^
        --embed_dim %EMBED_DIM% --activation %ACTI_F% --accumulation_steps %ACC_STEPS% ^
        --focus_graph %F_GRAPH% --normalization %NORM% --train_dataset !current_dataset! ^
        --optimizer %OPTIM% --hidden_dim %HIDDEN_DIM% --n_heads %N_HEADS% --dropout %DROPOUT% ^
        --waste_type %WTYPE% --focus_size %F_SIZE% --n_encode_layers %N_ENC_L% --lr_model %LR_MODEL% ^
        --eval_focus_size %VAL_F_SIZE% --distance_method %DIST_M% --exp_beta %EXP_BETA% ^
        --edge_threshold %EDGE_T% --edge_method %EDGE_M% --eval_batch_size %VAL_B_SIZE% ^
        --temporal_horizon %H_0% --lr_scheduler %LR_SCHEDULER% --lr_decay %LR_DECAY% ^
        --batch_size %B_SIZE% --lr_critic_value %LR_CV% --bl_alpha %BL_ALPHA% --area %AREA% ^
        --aggregation_graph %AGG_G% --dm_filepath "%DM_PATH%" ^
        --wandb_mode %WB_MODE% --distance_method %DM_METHOD% --mrl_method %META_METHOD% --mrl_lr %META_LR% ^
        --mrl_history %META_HISTORY% --mrl_step %META_STEP% --hrl_epochs %HRL_EPOCHS% --hrl_clip_eps %HRL_CLIP_EPS% ^
        --gat_hidden %GAT_HIDDEN% --lstm_hidden %LSTM_HIDDEN% --gate_prob_threshold %GATE_THRESH%

        if "%VERBOSE%"=="true" ( !CMD! ) else ( !CMD! >nul 2>&1 )
    )

    :: Note: You can replicate the pattern above for AMGC, TRANSGCN, DDAM, and TAM
    :: by changing the --model, --encoder, and the Horizon index (%H_1%, %H_2%, etc.)
)

echo.
echo ===== Training completed for all distributions =====
pause
