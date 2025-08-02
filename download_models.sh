how to download he 4.5 b model

export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download sand-ai/MAGI-1 \
  --local-dir ./downloads/t5_pretrained/t5-v1_1-xxl \
  --include "ckpt/t5/t5-v1_1-xxl/*" \
  --local-dir-use-symlinks False
huggingface-cli download sand-ai/MAGI-1 \
  --local-dir ./downloads/vae \
  --include "ckpt/vae/*" \
  --local-dir-use-symlinks False
huggingface-cli download sand-ai/MAGI-1 \
  --local-dir ./downloads/4.5B_base \
  --include "ckpt/magi/4.5B_base/*" \
  --local-dir-use-symlinks False
mv ./downloads/t5_pretrained/t5-v1_1-xxl/ckpt/t5/t5-v1_1-xxl/* ./downloads/t5_pretrained/t5-v1_1-xxl/
mv ./downloads/vae/ckpt/vae/* ./downloads/vae/
mv ./downloads/4.5B_base/ckpt/magi/4.5B_base/* ./downloads/4.5B_base/

huggingface-cli download sand-ai/MAGI-1 \
  --local-dir ./downloads/24B_base \
  --include "ckpt/magi/24B_base/*" \
  --local-dir-use-symlinks False
mv ./downloads/24B_base/ckpt/magi/24B_base/* ./downloads/24B_base/