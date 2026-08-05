#!/bin/bash
cd /Users/air/0805_img2vid_mps
mkdir -p batch_outputs
i=0
while IFS= read -r seed; do
  i=$((i+1))
  idx=$(printf '%02d' "$i")
  echo "=== [$(date '+%H:%M:%S')] Run $idx/10 seed=$seed ==="
  python3 img2vid_animatediff.py \
    --image 54736755_c057723f64.jpg \
    --output "batch_outputs/output_${idx}_seed${seed}.mp4" \
    --seed "$seed" \
    --frames 16 --fps 8 --steps 20
  echo "=== [$(date '+%H:%M:%S')] Done $idx/10 ==="
done < batch_outputs/seeds.txt
echo "BATCH_ALL_DONE"
