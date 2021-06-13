#! /usr/bin/env bash
cd /workspace/datasets/ISBI2021/BF-C2DL-HSC/ && python /opt/elephant/script/generate_seg_labels.py .
cd /workspace/datasets/ISBI2021/BF-C2DL-MuSC/ && python /opt/elephant/script/generate_seg_labels.py .
cd /workspace/datasets/ISBI2021/DIC-C2DH-HeLa/ && python /opt/elephant/script/generate_seg_labels.py --sparse /opt/elephant/script/DIC-C2DH-HeLa_sparse.json .
cd /workspace/datasets/ISBI2021/Fluo-C2DL-MSC/ && python /opt/elephant/script/generate_seg_labels.py --sparse /opt/elephant/script/Fluo-C2DL-MSC_sparse.json .
cd /workspace/datasets/ISBI2021/Fluo-N2DH-GOWT1/ && python /opt/elephant/script/generate_seg_labels.py --sparse /opt/elephant/script/Fluo-N2DH-GOWT1_sparse.json .
cd /workspace/datasets/ISBI2021/Fluo-N2DL-HeLa/ && python /opt/elephant/script/generate_seg_labels.py --sparse /opt/elephant/script/Fluo-N2DL-HeLa_sparse.json .
cd /workspace/datasets/ISBI2021/PhC-C2DH-U373/ && python /opt/elephant/script/generate_seg_labels.py .
cd /workspace/datasets/ISBI2021/PhC-C2DL-PSC/ && python /opt/elephant/script/generate_seg_labels.py .
