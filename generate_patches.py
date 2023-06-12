import argparse
import os
import pickle

from PIL import Image

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
parser.add_argument('--patch-size', type=int, default=64)
args = parser.parse_args()

max_im_w, max_im_h = 980, 644
patch_size = args.patch_size
images = []
for root, dirs, files in os.walk(args.data_path):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            images.append(os.path.join(root, file))


fragment_map = {}
for image_path in images:
    with Image.open(image_path) as f:
        image = f.convert('RGB')

    ratio = min(max_im_h / image.height, max_im_w / image.width)
    if ratio < 1:
        image = image.resize((int(ratio * image.width), int(ratio * image.height)), Image.LANCZOS)

    n_cols = image.width // patch_size
    n_rows = image.height // patch_size

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    patch_dir = os.path.join(args.output_path, image_name)
    os.makedirs(patch_dir, exist_ok=True)
    for i in range(n_cols):
        for j in range(n_rows):
            box = (j*patch_size, i*patch_size, (j+1)*patch_size, (i+1)*patch_size)
            patch = image.crop(box)
            patch_name = f'{j}_{i}.png'
            patch.save(os.path.join(patch_dir, patch_name))
            rel_path = os.path.join(image_name, patch_name)
            fragment_map.setdefault(image_name, []).append({'img': rel_path, 'col': j, 'row': i, 'name': image_name,
                                                            'positive': [], 'negative': []})

entries = {}
for image_name, fragments in fragment_map.items():
    for first in fragments:
        for second in fragments:
            if first['img'] == second['img']:
                continue
            if first['col'] == second['col'] and abs(first['row'] - second['row']) == 1:
                first['positive'].append(second)
            elif first['row'] == second['row'] and abs(first['col'] - second['col']) == 1:
                first['positive'].append(second)
            else:
                first['negative'].append(second)
        if len(first['positive']) > 0:
            entries.setdefault(image_name, []).append(first)
entry_map = {i: k for i, k in enumerate(entries.keys())}
entry_file = os.path.join(args.output_path, 'data.pkl')
with open(entry_file, 'wb') as f:
    pickle.dump({
        'entries': entries,
        'entry_map': entry_map
    }, f)
