#! /usr/bin/env bash
python /opt/elephant/script/generate_seg_labels.py --st --3d --cr 0.4 /workspace/datasets/ISBI2021/Fluo-C3DH-A549/
python /opt/elephant/script/generate_seg_labels.py --st --3d --cr 0.4 /workspace/datasets/ISBI2021/Fluo-C3DH-H157/
python /opt/elephant/script/generate_seg_labels.py --st --3d --cr 0.4 /workspace/datasets/ISBI2021/Fluo-C3DL-MDA231/
python /opt/elephant/script/generate_seg_labels.py --st --3d --cr 0.4 /workspace/datasets/ISBI2021/Fluo-N3DH-CE/
python /opt/elephant/script/generate_seg_labels.py --st --3d --cr 0.4 /workspace/datasets/ISBI2021/Fluo-N3DH-CHO/
