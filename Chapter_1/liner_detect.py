import cv2
import numpy as np

img_src = cv2.imread('picture/edge.png', 1)
img_dst = img_src.copy()
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)            #グレー画像にする

img_edge = cv2.Canny(img_gray, 200, 200)                        #キャニー法によるエッジ検出

#変数(入力画像,rhoの精度,thetaの精度,直線と見做されるのに必要な最低限の投票数)
lines = cv2.HoughLines(img_edge, 1, np.pi/180, 120)             #ハフ変換による直線検出
#返り値([rho,theta]の配列で返される,この場合は[44,1,2])

rows, cols = img_dst.shape[:2]                                  #[:2]は[4032, 3024, 3]を[4032, 3024]に変換すること

for rho, theta in lines[:, 0]:                                  #多次元配列の場合に使われ、全ての配列の0番目にアクセス
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    cv2.line(img_dst,                                            #線の描画
        (int(x0 - cols*(b)), int(y0 + cols*(a))),
        (int(x0 + cols*(b)), int(y0 - cols*(a))),
        (0, 0, 255), 2)

cv2.imwrite('picture/liner_detect.png', img_dst)

#rho:ρ, theta:θ
#直線が原点の下を通過する場合rhoは正、thetaは180度未満
#直線が原点の上を通過する場合rhoは負、thetaも負の値をとる
#x軸と平行な直線の場合、thetaは90度
#y軸と平行な直線の場合、thetaは0度
#配列のサイズは精度によって変化させる(1度ずつ調べる場合は180列必要)
#rhoは画像の直交方向の長さを最大値とする、1画素単位での精度が必要であれば、行の数を画像の対角方向の長さに設定