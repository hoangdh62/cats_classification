from imutils import paths
import cv2
from tqdm import tqdm
import os

DATASET_PATH = 'D:/Vinsofts/flowers_data/Combretum Indicum'
filePaths = list(paths.list_files(DATASET_PATH))
k = 1
for filePath in tqdm(filePaths):
    print(filePath)
    img = cv2.imread(filePath)
    # name = os.path.basename(filePath)
    H = img.shape[0]
    W = img.shape[1]
    crop_img = img[0:(H - H // 10), 0:W]
    cv2.imwrite("D:/Vinsofts/flowers_data/Combretum Indicum 2/"+str(k)+".jpg", crop_img)
    k+=1

