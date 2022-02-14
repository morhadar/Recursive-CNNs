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

def mesh_imgs(pil_images: list, grid_wh: tuple, titles=None):
    #todo - if grid is not given create the most squared grid floor(sqrt(N)) * ceil(sqrt(N)) ?
    assert grid_wh[0]*grid_wh[1] >= len(pil_images), 'grid is too small'
    titles = titles if titles is not None else ['']*len(pil_images)
    assert len(titles) == len(pil_images), 'not enough/too many titles'

    h_step = max([im.height for im in pil_images])
    w_step = max([im.width for im in pil_images])
    w = w_step * grid_wh[0]
    h = h_step * grid_wh[1]
    dst = Image.new('RGB', (w,h))
    d = ImageDraw.Draw(dst)

    coords_iterator = itertools.product(range(0, w, w_step), range(0, h, h_step)) #TODO - arrange by rows insted of colunms
    for i, im in enumerate(pil_images):
        p_start = next(coords_iterator)
        dst.paste(im, p_start)
        d.text(p_start, titles[i]) #TODO - control font size
    return dst

def draw_polygon_pil(im, quad, outline=None, width=1):
    quad_tmp = list(quad)
    quad_tmp.append(quad_tmp[0])
    ImageDraw.Draw(im).line(quad_tmp, fill=outline, width=width)
