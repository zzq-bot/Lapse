seed=(0 1 2 3 4)

for s in "${seed[@]}"
do
    python main.py --test-freq=80 --task="Ant-v2" --seed=${s} &
    python main.py --test-freq=80 --task="HalfCheetah-v2" --seed=${s} &
    python main.py --test-freq=80 --task="Hopper-v2" --seed=${s} &
    python main.py --test-freq=80 --task="Walker2d-v2" --seed=${s}
done
