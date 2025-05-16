seed=(0 1 2)
step=(6000 8000 12000 14000 16000 18000)

for t in "${step[@]}"
do
for s in "${seed[@]}"
do
python main.py --task="Ant-v2" --tag="cafes_len" --buffer-size=${t} --seed=${s} &
python main.py --task="HalfCheetah-v2" --tag="cafes_len" --buffer-size=${t} --seed=${s} &
python main.py --task="Hopper-v2" --trans-base="wocar" --tag="cafes_len" --buffer-size=${t} --seed=${s} &
python main.py --task="Walker2d-v2" --trans-base="wocar" --tag="cafes_len" --buffer-size=${t} --seed=${s}
done
done