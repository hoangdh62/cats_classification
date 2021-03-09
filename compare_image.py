from PIL import Image
import imagehash
from imutils import paths
from tqdm import tqdm
import os

i=0
imagePaths = list(paths.list_files('D:/Vinsofts/flowers_data/test'))
hash_bank = {}
hash_bank2 = {}
count = 0
for m, imagePath in enumerate(tqdm(imagePaths)):
    hash1 = imagehash.average_hash(Image.open(imagePath))
    if hash1 not in hash_bank:
        hash_bank[hash1] = imagePath
        continue
    else:
        hash_bank2[hash1] = imagePath
        print(imagePath)
        print(hash_bank[hash1])
        os.remove(imagePath)
    count += 1
print(count)