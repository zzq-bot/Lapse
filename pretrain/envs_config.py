import numpy as np


class Ant:
    mask = np.zeros((5, 111))
    mask[0, 0] = 1
    mask[1, 13:16] = 1
    mask[2, 1:27] = 1
    mask[2, 13:19] = 0
    mask[3, 16:19] = 1
    mask[4, 27:] = 1
    pair_idx = [[13, 14], [16, 17]]


class Halfcheetah:
    mask = np.zeros((5, 17))
    mask[0, 0] = 1
    mask[1, 8:10] = 1
    mask[2, 1:8] = 1
    mask[3, 10:17] = 1
    pair_idx = [[8, 9]]


class Hopper:
    mask = np.zeros((5, 11))
    mask[0, 0] = 1
    mask[1, 5:7] = 1
    mask[2, 1:5] = 1
    mask[3, 7:11] = 1
    pair_idx = [[5, 6]]


class Walker:
    mask = np.zeros((5, 17))
    mask[0, 0] = 1
    mask[1, 8:10] = 1
    mask[2, 1:8] = 1
    mask[3, 10:17] = 1
    pair_idx = [[8, 9]]


def get_unit_rtt_transform_vec(mask, stage: int):
    _half = 1 / 2
    _sqrt3 = _half * np.sqrt(3)
    rtt30 = np.array([[_sqrt3, -_half], [_half, _sqrt3]])
    rtt60 = np.array([[_half, -_sqrt3], [_sqrt3, _half]])
    rtt_vec = [None, rtt30, rtt30, rtt60, rtt60]
    rtt_vec += [None, None, None, None, None]
    rtt_vec *= 3
    rtt_vec = rtt_vec[stage - 1]

    _ft_scale = 3.281
    _rpm_scale = 9.549
    _deg_scale = 57.3
    _kmh_scale = 3.6
    # scales_mat = [
    #     [_ft_scale, _ft_scale, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [_ft_scale, _ft_scale, 1, _rpm_scale, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, _kmh_scale, 1, _rpm_scale, 1],
    # ]
    scales_mat = [
        [_ft_scale, _ft_scale, _ft_scale, _ft_scale, _ft_scale],
        [_ft_scale + 3, _ft_scale + 3, _ft_scale + 3, _ft_scale + 3, _ft_scale + 3],
        [_ft_scale + 6, _ft_scale + 6, _ft_scale + 6, _ft_scale + 6, _ft_scale + 6],
        [_ft_scale + 9, _ft_scale + 9, _ft_scale + 9, _ft_scale + 9, _ft_scale + 9],
        [_ft_scale + 12, _ft_scale + 12, _ft_scale + 12, _ft_scale + 12, _ft_scale + 12],
    ]
    scales_mat += [
        [_ft_scale + 12, _ft_scale + 12, _ft_scale + 12, _ft_scale + 12, _ft_scale + 12],
        [_ft_scale + 10, _ft_scale + 10, _ft_scale + 10, _ft_scale + 10, _ft_scale + 10],
        [_ft_scale + 8, _ft_scale + 8, _ft_scale + 8, _ft_scale + 8, _ft_scale + 8],
        [_ft_scale + 6, _ft_scale + 6, _ft_scale + 6, _ft_scale + 6, _ft_scale + 6],
        [_ft_scale + 4, _ft_scale + 4, _ft_scale + 4, _ft_scale + 4, _ft_scale + 4],
    ]
    scales_mat *= 3

    scale = np.array([scales_mat[stage - 1]]).T
    scale_vec = (mask * scale).sum(axis=0)

    return scale_vec, rtt_vec


def get_physics_func(env_id: str, stage: int):
    EnvClass = {"Ant-v2": Ant, "HalfCheetah-v2": Halfcheetah, "Hopper-v2": Hopper, "Walker2d-v2": Walker}[env_id]
    mask, pairs = EnvClass.mask, EnvClass.pair_idx
    scale_vec, rtt_vec = get_unit_rtt_transform_vec(mask, stage)

    def func(X, scale_vec=scale_vec, rtt_vec=rtt_vec, pairs=pairs):
        res = X * scale_vec
        if rtt_vec is not None:
            for pair in pairs:
                res[pair] = rtt_vec @ res[pair]
        return res

    return func


if __name__ == "__main__":
    mask = Halfcheetah.mask
    pairs = Halfcheetah.pair_idx

    I = np.ones(mask.shape[1])
    print(I)

    # env_id = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"][1]

    env_id = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"][1]

    # for stage in range(1, 6):
    #     func = get_physics_func(env_id, stage)
    #     res = func(I)
    #     print(res)

    for env in ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]:
        print(f"now env: {env}")
        EnvClass = {"Ant-v2": Ant, "HalfCheetah-v2": Halfcheetah, "Hopper-v2": Hopper, "Walker2d-v2": Walker}[env]
        mask, pairs = EnvClass.mask, EnvClass.pair_idx
        for i in range(1, 6):
            scale_vec, rtt_vec = get_unit_rtt_transform_vec(mask, i)
            print(scale_vec, rtt_vec)
