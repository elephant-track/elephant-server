#! /usr/bin/env python
# ==============================================================================
# Copyright (c) 2021, Ko Sugawara
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1.  Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Generate config files for the CTC 6th challenge (ISBI2021)."""
import argparse
from string import Template
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runtemplate', help='template file for run config')
    parser.add_argument('traintemplate', help='template file for train config')
    parser.add_argument('entrytemplate', help='template file for entry file')
    parser.add_argument('dir', help='output dir')
    args = parser.parse_args()
    scale_dict = {
        'BF-C2DL-HSC-01': '0.645, 0.645',
        'BF-C2DL-HSC-02': '0.645, 0.645',
        'BF-C2DL-MuSC-01': '0.645, 0.645',
        'BF-C2DL-MuSC-02': '0.645, 0.645',
        'DIC-C2DH-HeLa-01': '0.19, 0.19',
        'DIC-C2DH-HeLa-02': '0.19, 0.19',
        'Fluo-C2DL-MSC-01': '0.3, 0.3',
        'Fluo-C2DL-MSC-02': '0.3977, 0.3977',
        'Fluo-N2DH-GOWT1-01': '0.24, 0.24',
        'Fluo-N2DH-GOWT1-02': '0.24, 0.24',
        'Fluo-N2DL-HeLa-01': '0.645, 0.645',
        'Fluo-N2DL-HeLa-02': '0.645, 0.645',
        'PhC-C2DH-U373-01': '0.65, 0.65',
        'PhC-C2DH-U373-02': '0.65, 0.65',
        'PhC-C2DL-PSC-01': '1.6, 1.6',
        'PhC-C2DL-PSC-02': '1.6, 1.6',
        'Fluo-C3DH-A549-01': '0.126, 0.126, 1.0',
        'Fluo-C3DH-A549-02': '0.126, 0.126, 1.0',
        'Fluo-C3DH-H157-01': '0.126, 0.126, 0.5',
        'Fluo-C3DH-H157-02': '0.126, 0.126, 0.5',
        'Fluo-C3DL-MDA231-01': '1.242, 1.242, 6.0',
        'Fluo-C3DL-MDA231-02': '1.242, 1.242, 6.0',
        'Fluo-N3DH-CE-01': '0.09, 0.09, 1.0',
        'Fluo-N3DH-CE-02': '0.09, 0.09, 1.0',
        'Fluo-N3DH-CHO-01': '0.202, 0.202, 1.0',
        'Fluo-N3DH-CHO-02': '0.202, 0.202, 1.0',
    }
    datasets = [
        'BF-C2DL-HSC',
        'BF-C2DL-MuSC',
        'DIC-C2DH-HeLa',
        'Fluo-C2DL-MSC',
        'Fluo-N2DH-GOWT1',
        'Fluo-N2DL-HeLa',
        'PhC-C2DH-U373',
        'PhC-C2DL-PSC',
        'Fluo-C3DH-A549',
        'Fluo-C3DH-H157',
        'Fluo-C3DL-MDA231',
        'Fluo-N3DH-CE',
        'Fluo-N3DH-CHO',
    ]
    p = Path(args.dir)
    p_run = p / 'run_configs'
    p_run.mkdir(exist_ok=True, parents=True)
    p_train = p / 'train_configs'
    p_train.mkdir(exist_ok=True, parents=True)
    p_entry = p / 'entry_files'
    p_entry.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        for cfg in ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST'):
            if 'all' not in cfg:
                # train config
                with open(args.traintemplate, 'r') as f:
                    src = Template(f.read())
                result = src.substitute({
                    'dataset': dataset,
                    'config': cfg,
                })
                with open(str(p_train / f'{dataset}-{cfg}-seg.json'), 'w') as f:
                    f.write(result)
            if 'all' in cfg:
                modelbase = cfg
            else:
                modelbase = f'{dataset}-{cfg}'
            for i in range(2):
                seq = f'0{i+1}'
                basename = f'{dataset}-{seq}-{cfg}'

                # run config
                with open(args.runtemplate, 'r') as f:
                    src = Template(f.read())
                result = src.substitute({
                    'modelbase': modelbase,
                    'scales': scale_dict[f'{dataset}-{seq}'],
                    'is_3d': str('3D' in dataset).lower(),
                })
                with open(str(p_run / f'{basename}.json'), 'w') as f:
                    f.write(result)

                # entry files
                with open(args.entrytemplate, 'r') as f:
                    src = Template(f.read())
                result = src.substitute({
                    'dataset': dataset,
                    'seq': seq,
                    'config': cfg,
                })
                filename = str(p_entry / f'{basename}.sh')
                with open(filename, 'w') as f:
                    f.write(result)


if __name__ == '__main__':
    main()
