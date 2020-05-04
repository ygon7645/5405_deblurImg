import numpy as np
from PIL import Image
import click
import cv2 as cv
from model import generator_model
from utils import load_image, deprocess_image, preprocess_image


def deblur(image_path):
    data = {
        'A_paths': [image_path],
        'A': np.array([preprocess_image(load_image(image_path))])
    }
    x_test = data['A']
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=x_test)
    generated = np.array([deprocess_image(img) for img in generated_images])
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    print(generated.shape)
    #ima = Image.fromarray(generated)
    #dst = cv.filter2D(ima, -1, kernel=kernel)
    #dst.save("/content/drive/My Drive/5405_digitalMedia/result/e.png")
    #image_arr = np.array(dst)

    x_test = deprocess_image(x_test)
    '''
    img = generated[0, :, :, :]
    im = Image.fromarray(img.astype(np.uint8))
    im.save("/content/drive/My Drive/5405_digitalMedia/result/f.png")
    src = cv.imread("/content/drive/My Drive/5405_digitalMedia/result/f.png")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpen_image = cv.filter2D(src, cv.CV_32F, kernel=kernel)
    sharpen_image = cv.convertScaleAbs(sharpen_image)
    cv.imwrite("/content/drive/My Drive/5405_digitalMedia/result/g.png",sharpen_image)
    '''

    for i in range(generated_images.shape[0]):
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save("/content/drive/My Drive/5405_digitalMedia/result/i.jpg")#('deblur'+image_path)


@click.command()
@click.option('--image_path', help='Image to deblur')
def deblur_command(image_path):
    return deblur(image_path)


if __name__ == "__main__":
    deblur_command()
