seed=(0 1 2)
prm=(0.1 1.0 100.0 1000.0)
for p in "${prm[@]}"
do
for s in "${seed[@]}"
do
    python main.py--tag="param_cganlambda" --cgan-l2-lambda=${p} --task="Ant-v2" --seed=${s} &
    python main.py--tag="param_cganlambda" --cgan-l2-lambda=${p} --task="HalfCheetah-v2" --seed=${s} &
    python main.py--tag="param_cganlambda" --cgan-l2-lambda=${p} --task="Hopper-v2" --seed=${s} &
    python main.py--tag="param_cganlambda" --cgan-l2-lambda=${p} --task="Walker2d-v2" --seed=${s}
done
done


prm=(0.001 0.01 0.1 0.5)
for p in "${prm[@]}"
do
for s in "${seed[@]}"
do
    python main.py--tag="param_tau" --tau=${p} --task="Ant-v2" --seed=${s} &
    python main.py--tag="param_tau" --tau=${p} --task="HalfCheetah-v2" --seed=${s} &
    python main.py--tag="param_tau" --tau=${p} --task="Hopper-v2" --seed=${s} &
    python main.py--tag="param_tau" --tau=${p} --task="Walker2d-v2" --seed=${s}
done
done

prm=(0.8 0.9 0.95 0.999)
for p in "${prm[@]}"
do
for s in "${seed[@]}"
do
    python main.py--tag="param_gamma" --gamma=${p} --task="Ant-v2" --seed=${s} &
    python main.py--tag="param_gamma" --gamma=${p} --task="HalfCheetah-v2" --seed=${s} &
    python main.py--tag="param_gamma" --gamma=${p} --task="Hopper-v2" --seed=${s} &
    python main.py--tag="param_gamma" --gamma=${p} --task="Walker2d-v2" --seed=${s}
done
done

prm=(1.5 2.0 3.0 3.5)
for p in "${prm[@]}"
do
for s in "${seed[@]}"
do
    python main.py--tag="param_alphamax" --alpha-max=${p} --task="Ant-v2" --seed=${s} &
    python main.py--tag="param_alphamax" --alpha-max=${p} --task="HalfCheetah-v2" --seed=${s} &
    python main.py--tag="param_alphamax" --alpha-max=${p} --task="Hopper-v2" --seed=${s} &
    python main.py--tag="param_alphamax" --alpha-max=${p} --task="Walker2d-v2" --seed=${s}
done
done

prm=(0.1 0.3 0.7 0.9)
for p in "${prm[@]}"
do
for s in "${seed[@]}"
do
    python main.py--tag="param_adapttau" --adapt-tau=${p} --task="Ant-v2" --seed=${s} &
    python main.py--tag="param_adapttau" --adapt-tau=${p} --task="HalfCheetah-v2" --seed=${s} &
    python main.py--tag="param_adapttau" --adapt-tau=${p} --task="Hopper-v2" --seed=${s} &
    python main.py--tag="param_adapttau" --adapt-tau=${p} --task="Walker2d-v2" --seed=${s}
done
done