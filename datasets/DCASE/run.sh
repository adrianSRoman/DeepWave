#!/bin/sh

for file in /Volumes/T7/DCASE2022-Task3/DCASE2022_SELD_synth_data/mic/*.wav; do
        echo "Extracting ${file}"
        python3 ./extract_dataset.py --data="${file}" --metadata=/Volumes/T7/DCASE2022-Task3/DCASE2022_SELD_synth_data/metadata/;
done

# for idx_freq in 0 1 2 3 4 5 6 7 8; do
#     python3 ./../../scripts/merge_dataset.py --out=/Volumes/T7/DCASE2022-Task3/deepwave_dataset/D_freq${idx_freq}.npz /Volumes/T7/DCASE2022-Task3/deepwave_dataset/*_freq${idx_freq}_*.npz;
# done

# for idx_freq in 0 1 2 3 4 5 6 7 8; do
#     python3 ./../../scripts/train_crnn.py --dataset="/Volumes/T7/DCASE2022-Task3/deepwave_dataset/D_freq${idx_freq}.npz" --parameter="/Volumes/T7/DCASE2022-Task3/deepwave_dataset/D_freq${idx_freq}_train.npz" --D_lambda=0.10000000 --tau_lambda=0.10000000 --mu=0.9 --N_layer=5 --psf_threshold=0.00000100 --tanh_lin_limit=1.00000000 --loss=relative-l2 --tv_ratio=0.20000000 --lr=1e-08 --N_epoch=10 --batch_size=100 --seed=0;
# done

# Plotting intensity field
# python3 ../Pyramic/color_plot.py --datasets  /Volumes/T7/DCASE2022-Task3/deepwave_dataset/fold1_room10_mix001_freq[0-8]_3DMarco_singletrack.npz --parameters /Volumes/T7/DCASE2022-Task3/deepwave_dataset/D_freq[0-8]_train.npz --img_type=RNN --mode=disk --out=/Volumes/T7/DCASE2022-Task3/deepwave_dataset/color_plots_full_sky --show_catalog --lon_ticks='np.linspace(-180, 180, 5)'