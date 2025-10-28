import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import local_binary_pattern

def calc_metrics(image):
    # Calculate per channel mean and standard deviation
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # summary stats
    # filter out [0,0,0] pixels
    means = cv2.mean(lab)[:3]
    stddevs = cv2.meanStdDev(lab)[1].flatten()[:3]

    # local binary pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(grey, n_points, radius, method='uniform')

    # lbp metrics
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)

    return means, stddevs, lbp_mean, lbp_std

def visualise_metrics(means_seg, stddevs_seg, means_bg, stddevs_bg):
    # cat and whiskers plot for visualising metrics
    labels = ['L Channel', 'A Channel', 'B Channel']
    x = range(len(labels))
    fig, ax = plt.subplots()
    ax.bar([p - 0.2 for p in x], means_seg, width=0.4, label='Segmented Mean', color='b', alpha=0.6)
    ax.bar([p + 0.2 for p in x], means_bg, width=0.4, label='Background Mean', color='r', alpha=0.6)
    ax.errorbar([p - 0.2 for p in x], means_seg, yerr=stddevs_seg, fmt='o', color='b')
    ax.errorbar([p + 0.2 for p in x], means_bg, yerr=stddevs_bg, fmt='o', color='r')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pixel Value')
    ax.set_title('Color Metrics Comparison')
    ax.legend()
    plt.show()


fg_image = cv2.imread("fg_score.jpg", cv2.IMREAD_UNCHANGED)
fg_image = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)

image = cv2.imread("Bobby_Kangaroo_6.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# gamma filter to enhance contrast
image = exposure.adjust_log(image, 1)

_, fg_binary = cv2.threshold(fg_image, 80, 255, cv2.THRESH_BINARY)

segmented = cv2.bitwise_and(image, image, mask=fg_binary)
background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(fg_binary))

means_seg, stddevs_seg, lbp_mean_seg, lbp_std_seg = calc_metrics(segmented)
means_bg, stddevs_bg, lbp_mean_bg, lbp_std_bg = calc_metrics(background)

print("Background Region - Means (L, A, B):", means_bg)
print("Segmented Region - Means (L, A, B):", means_seg)
print("--------------------------------------------")
print("Background Region - Stddevs (L, A, B):", stddevs_bg)
print("Segmented Region - Stddevs (L, A, B):", stddevs_seg)
print("--------------------------------------------")
print("Segmented Region - LBP Mean:", lbp_mean_seg, "LBP Stddev:", lbp_std_seg)
print("Background Region - LBP Mean:", lbp_mean_bg, "LBP Stddev:", lbp_std_bg)

visualise_metrics(means_seg, stddevs_seg, means_bg, stddevs_bg)

# visualize
plt.imshow(segmented)
plt.axis('off')
plt.show()