import wikipediaapi
import os
import cv2

img = cv2.imread('plot.png')

labels = os.listdir('cat_data/train')
wiki_wiki = wikipediaapi.Wikipedia('en')
ny = wiki_wiki.page('Abyssinian cat')
# for label in labels:
#     try:
#         ny = wikipediaapi.Wikipedia.page(label)
#     except:
#         try:
#             ny = wikipediaapi.Wikipedia.page(label+' cat')
#         except:
#             print(label)
ff=1