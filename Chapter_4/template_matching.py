import cv2
import numpy as np

img_src = cv2.imread('picture/src.jpg', 1)
img_template = cv2.imread('picture/template.jpg', 1)

img_dst = img_src.copy()                                                                #値渡しのメソッドがcopyメソッド

h, w, ch = img_template.shape
img_minmax = cv2.matchTemplate(img_src, img_template, cv2.TM_CCOEFF_NORMED)             #テンプレ画像とソースの類似度を計算
#返り値(各位置での類似度を持った配列を返す

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_minmax)                            #類似度が最大と最小となる画素の位置を調べる
#返り値(最も類似度が低い値、最も類似度が高い値、最も類似度が低い場所、最も類似度が高い場所)

cv2.rectangle(img_dst, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 255, 255), 10)

cv2.imwrite('picture/detect.jpg', img_dst)

#max_locはx軸y軸の順番で格納されている
#cv2.TM_CCOEFF_NORMEDは正規化相互相関(内積計算の際の角度で類似度を判断)