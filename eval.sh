python cliport/eval.py model_task=packing-shapes \
                       eval_task=packing-shapes \
                       agent=ours \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=exps_ours \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False