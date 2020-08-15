import cv2
import numpy as np

img_src1 = cv2.imread('picture/example_before.jpg', 0)
img_src2 = cv2.imread('picture/example_after.jpg', 0)

detector = cv2.AKAZE_create()

kpts1, desc1 = detector.detectAndCompute(img_src1, None)
kpts2, desc2 = detector.detectAndCompute(img_src2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.match(desc1, desc2)

h1, w1 = img_src1.shape
h2, w2 = img_src2.shape

output = cv2.drawMatches(img_src1, kpts1, img_src2, kpts2, matches, None, flags = 2)
cv2.imwrite('picture/AKAZEでの特徴量検出.png', output)


#KAZEはAOS?と可変コンダクタンス拡散?を採用している->重要な特徴を残したままノイズ除去し、スケール不変性をもつ
#AKAZEはFED?使ってるから速い