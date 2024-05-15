instance=$1   # need to do this script on 20 different instances
#size=$2     # need to do for 4 8 16 32 64 128
#noise_level=$3    # need to do this for 0.0 0.01 0.03 0.1

target_type="random_parker_butterfly"

for size in 4 8 16 32 64 128
do
  for noise_level in 0.0 0.01 0.03 0.1
  do
    python src/scripts/butterfly_unknown_perm.py \
      --target_type $target_type \
      --size $size \
      --noise_level $noise_level \
      --alpha 1e-2 1e-1 1 1e1 1e2 \
      --n_iter 50 \
      --reduction "sum" \
      --n_guesses 5 \
      --cuda \
      --save_dir "results/butterfly_unknown_perm/target_type=${target_type}-size=${size}-noise_level=${noise_level}/instance=${instance}"
  done
done