# seed=42
# sample_dir="<your_test_sample_directory>"  # Replace with your actual sample directory
# test_ids="/project/outputs/ids/test.tsv"
# num_workers=8
# batch_size=16


# python3 /project/src/python/testing/compute_fid.py \
#       seed=${seed} \
#       sample_dir=${sample_dir} \
#       test_ids=${test_ids} \
#       batch_size=${batch_size} \
#       num_workers=${num_workers}  


# python3 /project/src/python/testing/flip_and_rotate.py --dataset_path_1 "/project/outputs/inference/.../actually_ddpm/.../PIL"

# python3 /project/src/python/testing/fid.py "<your_test_sample_directory>" "/project/outputs/inference/.../actually_ddpm/.../PIL" --normalize_images  --limit 99999999 --model "imagenet" --description real_vs_syn_imagenet_normalized

python3 "/project/src/python/testing/fid_ratio.py"
