import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def draw_in_row(fruits, sizes):
    images = map(Image.open, ['./images/'+fruit+'.jpg' for fruit in fruits])
    widths, heights = zip(*(i.size for i in images))

    unit_width = 200
    drawing_sizes = np.array(sizes*unit_width, dtype=int)
    total_width = drawing_sizes.sum()+unit_width
    max_height = drawing_sizes.max()

    new_im = Image.new('RGB', (total_width, max_height),color=(255,255,255,0))

    drawn_ball = False
    x_offset = 0
    for im, drawing_size in zip(images, drawing_sizes):
        if not drawn_ball and drawing_size >= unit_width:
            im_aux = Image.open('./images/tennis_ball.jpg')
            im_aux = im_aux.resize([unit_width, unit_width])
            new_im.paste(im_aux, (x_offset,0))
            drawn_ball = True
            x_offset += im_aux.size[0]
        im = im.resize([drawing_size,drawing_size])
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    ax = plt.figure(figsize=(7,5), dpi=80)
    plt.imshow(new_im)
    plt.axis('off')

if __name__ == '__main__':
    pass
