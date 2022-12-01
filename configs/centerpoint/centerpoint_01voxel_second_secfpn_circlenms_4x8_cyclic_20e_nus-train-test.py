_base_ = ['./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))

data_root = 'data/nuscenes/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + 'nuscenes_infos_train.pkl',),
    val=dict(
        ann_file=data_root + 'nuscenes_infos_train.pkl',),
    test=dict(
        ann_file=data_root + 'nuscenes_infos_train.pkl',))