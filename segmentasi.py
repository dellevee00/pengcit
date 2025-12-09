import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# 1. LOAD GAMBAR
# =========================
img1 = cv2.imread("gambar1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("gambar2.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise ValueError("Gambar tidak ditemukan. Pastikan img1.png dan img2.png ada di folder yang sama.")

# =========================
# 2. FUNGSI MENAMBAH NOISE
# =========================
def add_salt_pepper(image, prob=0.05):
    output = np.copy(image)
    rnd = np.random.rand(*image.shape)
    output[rnd < prob/2] = 0
    output[rnd > 1 - prob/2] = 255
    return output

def add_gaussian(image, sigma=20):
    gauss = np.random.normal(0, sigma, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Generate noise images
img1_sp = add_salt_pepper(img1)
img1_gs = add_gaussian(img1)

img2_sp = add_salt_pepper(img2)
img2_gs = add_gaussian(img2)

# =========================
# 3. METODE SEGMENTASI
# =========================
def roberts(img):
    k1 = np.array([[1, 0, -1],
                   [0, 0,  0],
                   [-1,0,  1]])

    k2 = np.array([[0, 1, 0],
                   [-1,0, 1],
                   [0, -1,0]])

    gx = cv2.filter2D(img, cv2.CV_64F, k1)
    gy = cv2.filter2D(img, cv2.CV_64F, k2)

    mag = np.sqrt(gx*2 + gy*2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def prewitt(img):
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    gx = cv2.filter2D(img, cv2.CV_64F, kx)
    gy = cv2.filter2D(img, cv2.CV_64F, ky)

    mag = np.sqrt(gx*2 + gy*2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def sobel(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt(gx*2 + gy*2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def frei_chen(img):
    r2 = np.sqrt(2)
    kx = np.array([[-1,0,1],[-r2,0,r2],[-1,0,1]])
    ky = np.array([[-1,-r2,-1],[0,0,0],[1,r2,1]])

    gx = cv2.filter2D(img, cv2.CV_64F, kx)
    gy = cv2.filter2D(img, cv2.CV_64F, ky)

    mag = np.sqrt(gx*2 + gy*2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Mapping metode
methods = {
    "Roberts": roberts,
    "Prewitt": prewitt,
    "Sobel": sobel,
    "Frei-Chen": frei_chen
}

# =========================
# 4. SIAPKAN FOLDER OUTPUT
# =========================
os.makedirs("hasil", exist_ok=True)

# =========================
# 5. PROSES SEMUA CITRA
# =========================
all_images = {
    "img1_original": img1,
    "img1_saltpepper": img1_sp,
    "img1_gaussian": img1_gs,

    "img2_original": img2,
    "img2_saltpepper": img2_sp,
    "img2_gaussian": img2_gs
}

for img_name, img in all_images.items():
    for method_name, method_func in methods.items():
        result = method_func(img)
        cv2.imwrite(f"hasil/{img_name}_{method_name}.png", result)

print("\n=== PROSES SELESAI ===")
print("Semua hasil disimpan di folder: hasil/")