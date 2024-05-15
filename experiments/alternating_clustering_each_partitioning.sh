target_type=$1  # "random_parker_butterfly" "dft"
noise_level=$2  # 0.0 0.1
size=$3

repeat=5
n_guesses=5
n_iter=50
save_dir="results/alternate_clustering_each_partitioning/target_type=${target_type}-noise_level=${noise_level}-size=${size}"

python src/scripts/alternating_clustering_each_partitioning.py \
  --target_type $target_type \
  --size $size \
  --noise_level $noise_level \
  --n_random_partition 1000 \
  --random_init \
  --alpha 1e-2 1e-1 1 1e1 1e2 \
  --n_iter $n_iter \
  --reduction "sum" \
  --n_guesses $n_guesses \
  --cuda \
  --save_dir "${save_dir}" \
  --repeat $repeat
