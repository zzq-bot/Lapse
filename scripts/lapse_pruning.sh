seed=(0 1 2 3 4)

for s in "${seed[@]}"
do
    python main.py --task="Ant-v2" --tag="cafes_pruning_new" --cafes-pruning=1 --pruning-val=0.2 --seed=${s} &
    python main.py --task="HalfCheetah-v2" --tag="cafes_pruning_new" --cafes-pruning=1 --pruning-val=0.2 --seed=${s} &
    python main.py --task="Hopper-v2" --tag="cafes_pruning_new" --cafes-pruning=1 --pruning-val=0.2 --seed=${s} &
    python main.py --task="Walker2d-v2" --tag="cafes_pruning_new" --cafes-pruning=1 --pruning-val=0.2 --seed=${s}
done
