#! /usr/bin/env bash
python /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_train_seg_versatile3d_001.json seg /opt/elephant/script/train_configs/versatile3d_001.json
python /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_train_seg_versatile3d_002.json seg /opt/elephant/script/train_configs/versatile3d_002.json
python /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_train_seg_versatile3d_003.json seg /opt/elephant/script/train_configs/versatile3d_003.json
python /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_train_seg_versatile3d_004.json seg /opt/elephant/script/train_configs/versatile3d_004.json
