from PIL import ImageDraw

def draw_circle_pil(im, xy, radious=5, fill=None, outline=None, width=1):
    x, y = xy #TODO - generalize to list of points
    d = int(1.4*radious) #~sqrt(2*radious**2)
    ImageDraw.Draw(im).ellipse((x-d, y-d, x+d, y+d), fill=fill, outline=outline, width=width)