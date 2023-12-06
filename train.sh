export CLIPORT_ROOT=$(pwd)

python cliport/train.py train.task=packing-shapes \
                        train.agent=ours \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=201000 \
                        train.exp_folder=exps_cjj_1 \
                        dataset.cache=False 