#! /usr/bin/env bash
ELEPHANT_PROFILE= kernprof -l -v -o ~/train.py.lprof /opt/elephant/script/train.py --baseconfig /opt/elephant/script/train_configs/base_profile_flow.json flow /opt/elephant/script/train_configs/profile_flow.json
