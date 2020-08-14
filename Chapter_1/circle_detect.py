import cv2
import numpy as np

img_src = cv2.imread('picture/moon.png', 1)
img_dst = img_src.copy()
img_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
img_temp = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
img_edge = cv2.convertScaleAbs(img_temp)

element8 = np.ones((3,3), np.uint8)
img_open = cv2.morphologyEx(img_edge, cv2.MORPH_OPEN, element8)
img_edge = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, element8)

#変数(入力画像,方法(cv2.HOUGH_GRADIENTのみ実装),キャニーエッジの引数の大きい方、円の中心を検出する際の投票数の閾値)
circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 50, 200)
#([1,33,3]、1階33行3列)

for x, y, r in circles[0, :]:
    cv2.circle(img_dst, (x,y), int(r), (0, 0, 255), 3)

cv2.imwrite('picture/moon_circle_detect.png', img_dst)


#エラーの理由がわからない:DeprecationWarning: an integer is required (got type numpy.float32).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
#うまく円を検出できず
