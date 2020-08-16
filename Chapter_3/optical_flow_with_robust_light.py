import cv2
import numpy as np

flow_w = 10
flow_h = 10

img_pre = cv2.imread('picture/example_before.jpg', 0)
img_now = cv2.imread('picture/example_after.jpg', 0)

#変数(入力画像1,入力画像2,フロー画像,各画像に対する画像ピラミッドのスケール(<1,0.5は隣り合う各層で、それぞれ半分のサイズ),画像ピラミッドの層の数,
#平均化ウィンドウサイズ,ピラミッドで行う反復回数,各ピクセルでのピクセル近傍領域のサイズ,ガウス分布の標準偏差,フラグ)
flow = cv2.calcOpticalFlowFarneback(img_pre, img_now, None, 0.5, 3, 30, 3, 5, 1.1, 0)       #物体が動いた差分の値？を返す
#返り値(480, 853, 2)でfloat32

rows, cols = img_now.shape[:2]

for y in range(0, cols, flow_h):
    for x in range(0, rows, flow_w):
        ps = (y, x)                                                                         #それぞれの画素をpsに格納
        pe = (ps[0] + int(flow[x][y][0]), ps[1] + int(flow[x][y][1]))                       #それぞれの画素にflowの値を格納
        cv2.line(img_now, ps, pe, (0, 0, 255), 2)

cv2.imwrite('picture/after_robust.png', img_now)

#平均化ウィンドウサイズ:値が大きい程、ノイズに強くなり高速な動きを検出。しかし、ボケたモーションフィールドになる
#各ピクセルでのピクセル近傍領域のサイズ:ピクセルで多項式展開を求めるために利用するサイズ。値が大きいと滑らかな表面で近似される。一般的に5or7
#標準偏差:多項式展開の基底として使われる導関数を滑らかにするガウス分布の標準偏差。5なら1.1, 7なら1.5が適当らしい