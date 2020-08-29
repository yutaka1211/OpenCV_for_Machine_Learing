import cv2
import numpy as np

win_src = 'src'
win_bin = 'bin'

cap = cv2.VideoCapture(0)
ret, img_src = cap.read()
h, w, ch = img_src.shape

div = 3
rct = (0, 0, int(w / div), int(h / div))

cv2.namedWindow(win_src, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_bin, cv2.WINDOW_NORMAL)

cri = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

while(True):
    th = 220
    ret, img_src = cap.read()
    bgr = cv2.split(img_src)
    ret, img_bin = cv2.threshold(bgr[2], th, 255, cv2.THRESH_BINARY)

    #変数(ヒストグラムの逆変換,テンプレ,停止基準)
    ret, rct = cv2.CamShift(img_bin, rct, cri)                          #矩形の新しい座標を取得
    #返り値(物体の位置、サイズ、姿勢、回転した矩形)これらを表現する構造体を返す

    pts = cv2.boxPoints(ret)                                            #矩形データ(構造体)を4点座標に変換
    pts = np.int0(pts)                                                  #整数にする
    cv2.polylines(img_src, [pts], True, 255, 2)                         #描画

    cv2.imshow(win_src, img_src)
    cv2.imshow(win_bin, img_bin)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#meanshiftに対して、ウィンドウサイズを調節したのがカムシフト
#CamShift:meanShift()を用いて物体の中心を求め、物体サイズに合わせて窓サイズを調節して、最適な方向を検出します