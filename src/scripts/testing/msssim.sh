seed=42
sample_dir="<your_test_sample_directory>"  # Replace with your actual sample directory
test_ids="/project/outputs/ids_lar/test.tsv"
num_workers=8
batch_size=16

python3 /project/src/python/testing/compute_msssim_sample.py\
       --seed=${seed} \
       --sample_dir=${sample_dir} \
       --test_ids=${test_ids} \
       --num_workers=${num_workers} 