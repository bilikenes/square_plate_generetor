import cv2
import numpy as np
from random import randint, uniform, choice, random
import os

def random_perspective(img, max_warp=0.18):
    h, w = img.shape[:2]
    def jitter(pt):
        x,y = pt
        return [x + uniform(-max_warp, max_warp)*w, y + uniform(-max_warp, max_warp)*h]
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([jitter((0,0)), jitter((w,0)), jitter((w,h)), jitter((0,h))])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def motion_blur(img, k=None):
    if k is None:
        k = randint(5, 15)
    kernel = np.zeros((k, k), dtype=np.float32)
    if random() < 0.5:
        kernel[k//2, :] = 1.0
    else:
        kernel[:, k//2] = 1.0
    angle = uniform(-25, 25)
    M = cv2.getRotationMatrix2D((k/2-0.5, k/2-0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / (kernel.sum() + 1e-6)
    return cv2.filter2D(img, -1, kernel)

def add_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def color_jitter(img, brightness=0.25, contrast=0.35, saturation=0.15):
    img = img.astype(np.float32) / 255.0
    b = uniform(-brightness, brightness)
    img = np.clip(img + b, 0, 1)
    c = 1.0 + uniform(-contrast, contrast)
    img = np.clip((img - 0.5) * c + 0.5, 0, 1)
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * (1.0 + uniform(-saturation, saturation)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def vignette(img, strength=0.4):
    h, w = img.shape[:2]
    X = cv2.getGaussianKernel(w, w*strength)
    Y = cv2.getGaussianKernel(h, h*strength)
    mask = (Y @ X.T)
    mask = mask / mask.max()
    mask = mask[..., None]
    mask = 0.3 + 0.7*mask
    return np.clip(img.astype(np.float32) * mask + (1-mask) * 255, 0, 255).astype(np.uint8)

def jpeg_compress(img, q=None):
    if q is None:
        q = randint(15, 40)
    enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def down_up(img):
    h,w = img.shape[:2]
    scale = uniform(0.35, 0.8)
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w,h), interpolation=choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))

def erode_ink(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (randint(1,3), randint(1,3)))
    bleed = cv2.dilate(bw, k, iterations=1)
    mask = cv2.cvtColor(bleed, cv2.COLOR_GRAY2RGB)/255.0
    return np.clip(img.astype(np.float32)*(1-mask) + np.minimum(img, 30)*(mask), 0, 255).astype(np.uint8)

def random_occlusions(img):
    h,w = img.shape[:2]
    out = img.copy()
    for _ in range(randint(1,3)):
        x = randint(3, w-4); y = randint(3, h-4)
        r = randint(2, 5)
        cv2.circle(out, (x,y), r, (randint(0,40),)*3, -1, lineType=cv2.LINE_AA)
    if random() < 0.1:
        x1,y1 = randint(0,w//2), randint(0,h-1)
        x2,y2 = randint(w//2,w-1), randint(0,h-1)
        thickness = randint(1,3)
        cv2.line(out, (x1,y1), (x2,y2), (randint(80,130),)*3, thickness, cv2.LINE_AA)
        out = cv2.GaussianBlur(out, (3,3), 0.8)
    return out

def subtle_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5.5,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def low_light(img, factor=None):
    if factor is None:
        factor = uniform(0.4, 0.6) 
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    out = add_noise(out, sigma=uniform(15,35))
    return out

def overexpose(img):
    img = img.astype(np.float32)
    mask = np.random.uniform(0.7, 1.0, img.shape[:2])[...,None]
    out = img + 180*mask  # highlight glare
    return np.clip(out,0,255).astype(np.uint8)

def add_shadow(img):
    h,w = img.shape[:2]
    mask = np.ones((h,w), np.float32)
    x1, y1 = randint(0,w//2), 0
    x2, y2 = randint(w//2,w), h
    poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32)
    shadow = np.zeros((h,w), np.float32)
    cv2.fillPoly(shadow, [poly], 0.5)  # %50 koyuluk
    mask = np.minimum(mask, shadow+0.5)
    return np.clip(img.astype(np.float32) * mask[...,None], 0, 255).astype(np.uint8)

def add_fog(img, strength=0.4):
    h,w = img.shape[:2]
    fog = np.full((h,w,3), 255, np.uint8)
    alpha = cv2.GaussianBlur(np.random.rand(h,w).astype(np.float32),
                             (w//3|1, h//3|1), 0)
    alpha = (alpha/alpha.max()) * strength
    return np.clip(img*(1-alpha[...,None]) + fog*alpha[...,None], 0,255).astype(np.uint8)

def chromatic_aberration(img, shift=2):
    b,g,r = cv2.split(img)
    M1 = np.float32([[1,0,shift],[0,1,shift]])
    r = cv2.warpAffine(r, M1, (img.shape[1], img.shape[0]))
    M2 = np.float32([[1,0,-shift],[0,1,-shift]])
    b = cv2.warpAffine(b, M2, (img.shape[1], img.shape[0]))
    return cv2.merge([b,g,r])


def environment_effects(img):
    funcs = [low_light, overexpose, add_shadow, add_fog, chromatic_aberration]
    if random() < 0.7:
        func = choice(funcs)
        return func(img)
    return img


def degrade_pipeline(img):
    img = img.copy()
    img = motion_blur(img, k=randint(1,3))
    img = add_noise(img, sigma=uniform(8,18))
    img = down_up(img)
    img = erode_ink(img)
    img = environment_effects(img)
    img = vignette(img, strength=uniform(0.4,0.9))
    img = jpeg_compress(img, q=randint(18,35))
    img = np.array(255 * (img / 255) ** 2.3, dtype='uint8')
    return img

input_folder = r"D:\Medias\normal_plates\TR\result"   
output_folder = r"D:\Medias\normal_plates\TR\result_1" 
os.makedirs(output_folder, exist_ok=True)
count = 0

for fname in os.listdir(input_folder):
    in_path = os.path.join(input_folder, fname)
    out_path = os.path.join(output_folder, fname)

    img = cv2.imread(in_path)

    degraded = degrade_pipeline(img)

    new_w = randint(110, 130)
    new_h = randint(60, 70)
    degraded = cv2.resize(degraded, (new_w, new_h), interpolation=cv2.INTER_AREA)

    degraded = cv2.GaussianBlur(degraded, (3, 3), 0)
    noise = np.random.normal(0, 15, degraded.shape).astype(np.int16)
    degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    out_jpg = os.path.splitext(out_path)[0] + ".jpg"
    cv2.imwrite(out_jpg, degraded, [int(cv2.IMWRITE_JPEG_QUALITY), randint(65, 75)])
    print(count)
    count += 1