#! /usr/bin/env bash
ELEPHANT_PROFILE= kernprof -l -v -o ~/train.py.lprof /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_profile_seg.json seg /opt/elephant/script/train_configs/profile_seg.json
# ELEPHANT_PROFILE= python -m memory_profiler /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_profile_seg.json seg /opt/elephant/script/train_configs/profile_seg.json
