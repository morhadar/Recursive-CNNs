import itertools
from PIL import Image, ImageDraw

def draw_circle_pil(im, xy, radious=5, fill=None, outline=None, width=1):
    x, y = xy #TODO - generalize to list of points
    d = int(1.4*radious) #~sqrt(2*radious**2)
    ImageDraw.Draw(im).ellipse((x-d, y-d, x+d, y+d), fill=fill, outline=outline, width=width)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def rr():
    print('gg')

def mesh_imgs(pil_images: list, grid_wh: tuple):
    assert grid_wh[0]*grid_wh[1] >= len(pil_images), 'grid is too small'

    h_step = max([im.height for im in pil_images])
    w_step = max([im.width for im in pil_images])
    w = w_step * grid_wh[0]
    h = h_step * grid_wh[1]
    dst = Image.new('RGB', (w,h))

    p_start = itertools.product(range(0, w, w_step), range(0, h, h_step))
    #TODO - arrange by rows insted of colunms
    for i, im in enumerate(pil_images):
        dst.paste(im, next(p_start))
    return dst
