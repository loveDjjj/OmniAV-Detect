export USE_AUDIO_IN_VIDEO=True
export use_audio_in_video=True
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=29511
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --model /data/OneDay/models/qwen/Qwen2.5-Omni-7B \
  --model_type qwen2_5_omni \
  --tuner_type lora \
  --dataset /data/OneDay/OmniAV-Detect/data/swift_sft/mavosdd/mavosdd_binary_train.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --dataloader_num_workers 8 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir /data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_mavosdd_binary_lora_audio_in_video