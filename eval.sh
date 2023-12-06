export CLIPORT_ROOT=$(pwd)

python cliport/eval.py model_task=packing-shapes \
                       eval_task=packing-shapes \
                       agent=ours \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=exps_cjj_1 \
                       update_results=True \
                       disp=False \
                       model_path=/home/shinji106/ntu/cliport/exps_cjj_1/packing-shapes-ours-n1000-train/checkpoints/