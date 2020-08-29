import cv2
import numpy as np

win_src = 'src'
win_bin = 'bin'

#変数(撮影に使うカメラのデバイス番号)
cap = cv2.VideoCapture(0)

#変数(データの読み込み可否,映像のパラメータの取得)
ret, img_src = cap.read()
h, w, ch = img_src.shape

div = 4
rct = (0, 0, int(w / div), int(h / div))


#変数(ウィンドウの名前、ウィンドウの設定(_NORMALにすると任意で大きさを決めれる))
cv2.namedWindow(win_src, cv2.WINDOW_NORMAL)                                     #ウィンドウの設定
cv2.namedWindow(win_bin, cv2.WINDOW_NORMAL)

#変数(指定された繰り返しの最大回数、指定された精度(epsilon)、回数は10、精度は1)
cri = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)                  #繰り返し処理の終了条件

while(True):
    th = 220
    ret, img_src = cap.read()
    bgr = cv2.split(img_src)
    ret, img_bin = cv2.threshold(bgr[2], th, 255, cv2.THRESH_BINARY)

    #変数(物体のヒストグラムの逆投影,テンプレの初期状態,停止基準)
    ret, rct = cv2.meanShift(img_bin, rct, cri)
    x, y, w, h = rct
    img_src = cv2.rectangle(img_src, (x, y), (x + w, y + h), (255, 0, 0), 3)    #検出の長方形描画

    cv2.imshow(win_src, img_src)
    cv2.imshow(win_bin, img_bin)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#meanshift:物体の反復探索アルゴ反復数が規定回数に達するか、テンプレの移動距離がeps以下になるまで動作する