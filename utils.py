from PIL import Image
from skimage.measure import structural_similarity as ssim
import cv2

def convert_image(image_name, num):
    image = Image.open(image_name)
    if num == 0:
        image_name = image_name[:3] + '.bmp'
    elif chromo[i][j] == 1:
        image_name = image_name[:3] + '.gif'
    elif chromo[i][j] == 2:
        image_name = image_name[:3] + '.jpg'
    elif chromo[i][j] == 3:
        image_name = image_name[:3] + '.png'
    image.save(image_name)


def ssim_score(im_name1, im_name2):
    im1 = cv2.imread(im_name1)
    im2 = cv2.imread(im_name2)

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    score = ssim(im1, im2)

    return score
