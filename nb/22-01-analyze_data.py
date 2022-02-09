# %%
import os
import subprocess

def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
os.chdir(getGitRoot())
print(os.getcwd())

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
# %% #! the required data for network1('document'-resnet20(8)) traninig:
gt_file = 'data/smartDocData_DocTestC/gt.csv'
gt = pd.read_csv('data/smartDocData_DocTestC/gt.csv', header=None)
d = os.path.dirname(gt_file)

i = Image.open(f'{d}/{gt.iloc[0,0]}')
t = np.array([[0.5891,0.1573],[0.8659,0.2978],[0.8227,0.7955],[0.4801,0.6449]])
w,h = i.size
t[:,0] *= w
t[:,1] *= h
i = np.array(i)
cv2.circle(i, t[0], 20, (255, 255, 127), 20)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(i)
# Image.fromarray(i).show()
# cv2.imwrite(f'{args.outputPath}/{img_name}{suffix}.jpg', im_out)

# im_out = draw_points((tl, tr, br, bl), img_orig)
# cv2.line(im_out, tl, tr, (0, 0, 255), 2)
# cv2.line(im_out, tr, br, (0, 0, 255), 2)
# cv2.line(im_out, br, bl, (0, 0, 255), 2)
# cv2.line(im_out, bl, tl, (0, 0, 255), 2)

# %% #! debug vizualize.py

from utils import mesh_imgs
from PIL import ImageDraw

im1 = Image.new('RGB', (200, 200), (228, 150, 150)); ImageDraw.Draw(im1).text((10,10), '1')
im2 = Image.new('RGB', (200, 200), (228, 255, 255)); ImageDraw.Draw(im2).text((10,10), '2')
im3 = Image.new('RGB', (200, 200), (228, 0, 0)); ImageDraw.Draw(im3).text((10,10), '3')
im4 = Image.new('RGB', (200, 200), (0, 128, 0)); ImageDraw.Draw(im4).text((10,10), '4')
imgs = [im1, im2, im3, im4]

big_image_pil1 = mesh_imgs(imgs, grid_wh=[2,2])
big_image_pil1

im1 = Image.new('RGB', (200, 100), (228, 150, 150)); ImageDraw.Draw(im1).text((10,10), '1')
im2 = Image.new('RGB', (150, 170), (0, 0, 255)); ImageDraw.Draw(im2).text((10,10), '2')
im3 = Image.new('RGB', (222, 346), (228, 0, 0)); ImageDraw.Draw(im3).text((10,10), '3')
im4 = Image.new('RGB', (120, 200), (0, 150, 0)); ImageDraw.Draw(im4).text((10,10), '4')
im5 = Image.new('RGB', (200, 200), (150, 150, 150)); ImageDraw.Draw(im5).text((10,10), '5')
im6 = Image.new('RGB', (300, 200), (255, 255, 255)); ImageDraw.Draw(im6).text((10,10), '6')
im7 = Image.new('RGB', (200, 300), (228, 150, 150)); ImageDraw.Draw(im7).text((10,10), '7')
imgs = [im1, im2, im3, im4, im5, im6, im7]

big_image_pil2 = mesh_imgs(imgs, grid_wh=[2,4])
big_image_pil2
