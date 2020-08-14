import cv2

img_src = cv2.imread('picture/src1.png', 1)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
#変数(入力画像,エッジの接続(2つの内小さい値),初期セグメントの検出)
img_dst = cv2.Canny(img_gray, 75, 300)

cv2.imwrite('picture/src1_canny.png', img_dst)