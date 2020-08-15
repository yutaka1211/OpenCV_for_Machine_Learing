import cv2
import numpy as np

img_src1 = cv2.imread('picture/example_before.jpg', 0)
img_src2 = cv2.imread('picture/example_after.jpg', 0)

detector = cv2.xfeatures2d.SIFT_create()                                #SIFT特徴量検出を始める
#detector = cv2.xfeatures2d.SURF_create()                               #SURF特徴量検出を始める

kpts1, desc1 = detector.detectAndCompute(img_src1, None)                #特徴量検出
kpts2, desc2 = detector.detectAndCompute(img_src2, None)

matcher = cv2.BFMatcher()                                               #特徴点の特徴点記述子を計算し
matches = matcher.match(desc1, desc2)                                   #総当たり(ブルートフォース)でマッチング
#返り値(列順にリストが格納されてる)

h1, w1 = img_src1.shape
h2, w2 = img_src2.shape

output = cv2.drawMatches(img_src1, kpts1, img_src2, kpts2, matches, None, flags = 2)
cv2.imwrite('picture/SIFTでの特徴量検出.png', ouput)

#SIFT(Scale-invariant feature transform)は特徴点の検出と特徴量の記述が行える
#拡大縮小、回転、照明変化に強い
#DoG画像をベースに特徴てんを検出->ロバストだが、画像の平滑化によって特徴が消える可能性がある
#kpts:keypoints,desc1:descriptors
#drawMatchesのflagsの引数,0:デフォルト,1:出力画像にも描画(出力できず),2:single_pointsには描画しない,4:rich_keypoints(特徴点の特徴強度:時計みたい)を描画