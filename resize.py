from PIL import Image, ImageFile
import glob
from tqdm import tqdm
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

print('starting')
train_dir = 'data/MIMIC-CXR-JPG/train_images'
test_dir = 'data/MIMIC-CXR-JPG/test_images'

train_resized_dir = 'data/MIMIC-CXR-JPG/train_images_resized'
test_resized_dir = 'data/MIMIC-CXR-JPG/test_images_resized'



def resize_images(dir, name, resized_path):
    basewidth = 512
    path = f'{dir}/{name}'
    img = Image.open(path)

    wpercent = (basewidth/float(img.size[0]))
    
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize))
    
    img.save(f'{resized_path}/{name}')


train_listdir = os.listdir(train_dir)
test_listdir = os.listdir(test_dir)

print('Processing training data................')
for train_name in tqdm(train_listdir):
    resize_images(train_dir, train_name, train_resized_dir)

print('Processing testing data................')
for test_name in tqdm(test_listdir):
    resize_images(test_dir, test_name, test_resized_dir)

