# 5e5 it @ 35 it/s ~ 4 h
echo "==EXPERIMENT START== sphere/base"
python3 run_model.py --mode=train --model=cloth --checkpoint_dir="$DATA"/chk_sphere_base --dataset_dir="$DATA"/sphere_simple \
  --num_training_steps=500_000
# 5e5 it @ 35 it/s ~ 4 h
echo "==EXPERIMENT START== sphere/model"
python3 run_model.py --mode=train --model=cloth --checkpoint_dir="$DATA"/chk_sphere_model --dataset_dir="$DATA"/sphere_simple \
  --num_training_steps=500_000 --subeq_model
echo "==EXPERIMENT START== sphere/layers"
# 2e5 it @ 5.5 it/s ~ 9.4 h
python3 run_model.py --mode=train --model=cloth --checkpoint_dir="$DATA"/chk_sphere_layers --dataset_dir="$DATA"/sphere_simple \
  --num_training_steps=200_000 --subeq_layers --subeq_encoder

# 5e5 it @ 36 it/s ~ 4 h
echo "==EXPERIMENT START== flag/base"
python3 run_model.py --mode=train --model=cloth --checkpoint_dir="$DATA"/chk_flag_base --dataset_dir="$DATA"/flag_simple \
  --num_training_steps=500_000
# 5e5 it @ 36 it/s ~ 4 h
echo "==EXPERIMENT START== flag/model"
python3 run_model.py --mode=train --model=cloth --checkpoint_dir="$DATA"/chk_flag_model --dataset_dir="$DATA"/flag_simple \
  --num_training_steps=500_000 --subeq_model
# 2e5 it @ 5.9 it/s ~ 9.4 h
echo "==EXPERIMENT START== flag/layers"
python3 run_model.py --mode=train --model=cloth --checkpoint_dir="$DATA"/chk_flag_layers --dataset_dir="$DATA"/flag_simple \
  --num_training_steps=200_000 --subeq_layers --subeq_encoder

# Total runtime: ~35h @ 0.43$/h ~ 15$


