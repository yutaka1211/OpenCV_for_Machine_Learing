import cv2
import numpy as np

flow_w = 10
flow_h = 10

img_pre = cv2.imread('picture/example_before.jpg', 0)
img_now = cv2.imread('picture/example_after.jpg', 0)

rows, cols = img_now.shape

ps = np.empty((0, 2), np.float32)                                                   #初期化せずに配列を作成し、空の配列を作成

#変数(開始,終了,増分)
for y in range(0, cols, flow_h):
    for x in range(0, rows, flow_w):
        pp = np.array([[y, x]], np.float32)
        ps = np.vstack((ps, pp))

"""これにより
ps = ([0., 0.],
      [0., 10.],
      [0., 20.],
      [0., 30.],
      ・・・
      [850., 470.]
      )
と追加されていく
"""

#変数(入力画像1,入力画像2,フローを検出する必要がある点のベクトル)
pe, status, error = cv2.calcOpticalFlowPyrLK(img_pre, img_now, ps, None)            #画像ピラミッドを利用して、Lucas-Kanade法を反復実行し、疎な特徴集合に対するオプティカルフローの作成

for i in range(len(ps)):
    cv2.line(img_now, (ps[i][0], ps[i][1]), (pe[i][0], pe[i][1]), (0, 0, 255), 2)

cv2.imwrite('picture/after.png', img_now)

#np.emptyを使うことでnp.zerosやnp.onesよりも高速
#np.empty(0, 2)の使い方は、配列を結合したいが、最終的に得られる配列のサイズがわからない場合や、計算規模が小さくプログラムの可読性を重視したい場合に使用される
#vstack(vertical stack)配列の左側が異なる時、使用できる
#hstack(horizontal stack)配列の左側以外が異なる時、使用できる