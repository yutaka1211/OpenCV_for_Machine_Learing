import cv2

img_src = cv2.imread('picture/src1.png', 1)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)            #グレースケールへ
img_dst = img_src.copy()                                        #値渡しする(ポインタで渡してない)

#変数(入力画像,検出したいコーナーの数、コーナーの最低限の質を指定するレベルを0~1で指定,検出される2つのコーナ間の最低限のユークリッド距離を与える)
corners = cv2.goodFeaturesToTrack(img_gray, 1000, 0.1, 5)
#スコアが指定されたレベルより低いコーナーが取り除かれ、残ったコーナーは、スコアを基に降順でソートされ、最もスコアの大きいコーナーの周囲のコーナーを除いて出力

for i in corners:
    x,y = i.ravel()                                             #x = i[0], y = i[1]とほぼ同じ意味
    #変数(入力画像,中心の座標,半径,描画する色,太さ)
    cv2.circle(img_dst, (int(x), int(y)), 3, (0, 0, 255), thickness = 2)

cv2.imwrite('picture/src1_corner_detect.png', img_dst)

#goodFeaturesToTrackはShi-Tomasiによるコーナー検出:R=min(λ1,λ2),このRの値が閾値より大きればコーナーとする
#ravel()は多次元のリストを1次元のリストにして返す、ポインタ的に渡すので変更後は値も変わる
#flatten()も同様の昨日だが、値渡しより、変更後は値が変わらない