#!/usr/bin/env python
import argparse
from pathlib import Path

import skimage.io
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='root path of CTC dataset')
    parser.add_argument(
        '--3d',
        dest='is_3d',
        action='store_true',
        help=('specify if generate 3D labels (only used for 3D+time datasets)')
    )
    parser.add_argument(
        '--gt',
        dest='gt',
        action='store_true',
        help=('specify if process GT')
    )
    parser.add_argument(
        '--st',
        dest='st',
        action='store_true',
        help=('specify if process ST')
    )
    parser.add_argument(
        '--gtst',
        dest='gtst',
        action='store_true',
        help=('specify if process GT+ST')
    )
    args = parser.parse_args()
    ref_types_list = []
    if args.gt:
        ref_types_list.append(('GT',))
    if args.st:
        ref_types_list.append(('ST',))
    if args.gtst:
        ref_types_list.append(('GT', 'ST'))
    if len(ref_types_list) == 0:
        ref_types_list = [('GT',), ('ST',), ('GT', 'ST')]
    p = Path(args.dir)
    is_3d = args.is_3d
    seg_dir_name = f'SEG{"_3d" if is_3d else ""}'
    for i in range(2):
        p_img = p / f'0{i+1}'
        shape = None
        dtype = None
        for f in tqdm(sorted(p_img.glob('*.tif'))):
            img = skimage.io.imread(str(f))
            if shape is None:
                shape = img.shape
                dtype = img.dtype
            else:
                if shape != img.shape:
                    print(f'image {f.name}: shape expected {shape} but got {img.shape}.')
                if dtype != img.dtype:
                    print(f'image {f.name}: dtype expected {dtype} but got {img.dtype}.')
        for ref_types in ref_types_list:
            for ref_type in ref_types:
                p_seg = p / f'0{i+1}_{ref_type}' / seg_dir_name
                if p_seg.exists():
                    for f in tqdm(sorted(p_seg.glob('*.tif'))):
                        img = skimage.io.imread(str(f))
                        if shape != img.shape:
                            print(f'image {f.name}: shape expected {shape} but got {img.shape}.')

if __name__ == "__main__":
    main()