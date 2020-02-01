from PIL import Image,ImageDraw
# import PIL as pil

def draw_bb(pimg,bb_array,pid2vid,show_vid,color=(0,255,0),text=''):
    """

    :rtype: Image
    :type pimg: Image
    """
    pimg_draw = ImageDraw.Draw(pimg)
    for i in range(len(bb_array)):
        pimg_draw.rectangle([bb_array[i,0],bb_array[i,1],bb_array[i,2],bb_array[i,3]],width=2,outline=color)
        if not pid2vid is None:
            pimg_draw.text(xy=[bb_array[i,0], bb_array[i,1]],
                     text='%d' % pid2vid[i], fill=(255,0,0))
        pimg_draw.text(xy=[bb_array[i, 0], bb_array[i, 3]],
                       text='%s' % text, fill=color)
    if show_vid:
        pimg.show()
    return pimg
def draw_bb_numpy_img(img,bb_array,pid2vid,show_vid,color=(0,255,0),text=''):
    """

    :type pimg: Image
    """
    pimg = Image.fromarray(img.astype('uint8'))
    if bb_array is None:
        return pimg
    pimg_draw = ImageDraw.Draw(pimg)
    for i in range(len(bb_array)):
        pimg_draw.rectangle([bb_array[i,0],bb_array[i,1],bb_array[i,2],bb_array[i,3]],width=2,outline=color)
        if not pid2vid is None:
            pimg_draw.text(xy=[bb_array[i,0], bb_array[i,1]],
                     text='%d' % pid2vid[i], fill=(255,0,0))
        pimg_draw.text(xy=[bb_array[i, 0], bb_array[i, 3]],
                       text='%d' % bb_array[i,-2], fill=color)
    if show_vid:
        pimg.show()
    return pimg