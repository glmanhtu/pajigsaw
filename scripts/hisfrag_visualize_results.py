import argparse
import csv
import os
import random

import pandas as pd
import torchvision.transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.utils import make_grid

parser = argparse.ArgumentParser("Prediction visualization", add_help=True)
parser.add_argument("--dataset-dir", required=True, help="Path to the images dataset")
parser.add_argument("--distance_matrix", required=True, help="Path to edges json file")

args = parser.parse_args()

with open(args.distance_matrix) as csv_file:
    # creating an object of csv reader
    # with the delimiter as ,
    csv_reader = csv.reader(csv_file, delimiter=',')

    # list to store the names of columns
    list_of_column_names = []

    # loop to iterate through the rows of csv
    for row in csv_reader:
        # adding the first row
        list_of_column_names = row

        # breaking the loop after the
        # first iteration itself
        break


def read_img(img_name, is_correct, border_size=10):
    cropper = torchvision.transforms.CenterCrop(512 - border_size * 2 - 4)
    img_file = os.path.join(args.dataset_dir, img_name + ".jpg")
    with Image.open(img_file) as f:
        img = f.convert('RGB')
    img = cropper(img)
    if border_size > 0:
        border_color = 'green' if is_correct else 'red'
        img = ImageOps.expand(img, border=(border_size, border_size, border_size, border_size), fill=border_color)
    img = ImageOps.expand(img, border=(2, 2, 2, 2), fill='white')

    return torchvision.transforms.ToTensor()(img)


n_col = 7
n_items = 9
column_idxs = random.sample(range(1, len(list_of_column_names)), k=n_col)

# Load similarity matrix from CSV file
distance_matrix = pd.read_csv(args.distance_matrix, index_col=0, usecols=[0] + column_idxs)
similarity_matrix = 1 - distance_matrix
images = []
for col in column_idxs:
    col_images = []
    col_name = list_of_column_names[col]
    print('col_name', col_name)
    author = col_name.split("_")[0]
    records = similarity_matrix[col_name].nlargest(n_items)
    col_images.append(read_img(col_name, is_correct=True, border_size=0))
    for key, value in records.items():
        record_author = key.split("_")[0]
        col_images.append(read_img(key, record_author == author))
    images.append(col_images)

grid_images = []
for i in range(len(images[0])):
    if i == 1:
        continue
    for j in range(len(images)):
        grid_images.append(images[j][i])

# make grid from the input images
# this grid contain 2 rows and 3 columns
Grid = make_grid(grid_images, nrow=n_col)

# display result
img = torchvision.transforms.ToPILImage()(Grid)
img.show()

