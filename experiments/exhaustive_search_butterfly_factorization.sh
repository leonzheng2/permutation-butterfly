size=$1

python src/scripts/exhaustive_search_butterfly_factorization.py \
  --size $size \
  --save_dir "results/exhaustive_search_butterfly_factorization/size=${size}"
