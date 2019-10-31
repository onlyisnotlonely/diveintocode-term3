import argparse
import cv2
import numpy as np
import os
import sys
import time
from keras.models import load_model
# ライブラリまでのディレクトリ定義
sys.path.append('./libs')
from init_camera import init_camera, get_image
from detect import detect
from classify import categorical_pred, binary_pred


bottle_str = 'bottle'
desc_text = 'Please place the bottle in the blue box.'
push_text = 'Please press "Enter"'
scan_text = 'Scaning...'
img_path = 'tmp/data.jpg'
font = cv2.FONT_HERSHEY_SIMPLEX
size = 1
weight = 2
color = (255,255,255)
text_coord = (10, 30)
box_color = (255, 0, 0)

parser = argparse.ArgumentParser()
parser.add_argument("--prototxt", default="data/mobilenet_v2_deploy.prototxt")
parser.add_argument("--caffemodel", default="models/mobilenet_v2.caffemodel")
parser.add_argument("--classNames", default="data/synset.txt")
parser.add_argument("--raspi", default=True)
parser.add_argument("--auto", default=False)

args = parser.parse_args()
raspi_flg = (args.raspi == True or args.raspi == "True" or args.raspi == "true")
auto_flg = (args.auto == True or args.auto == "True" or args.auto == "true")
f = open(args.classNames, 'r')
class_names = f.readlines()
cap = init_camera(raspi_flg)

# モデル読み込み
if raspi_flg:
    categorical_model = load_model('models/MobileNetV2_shape224_six_classes.h5')
    top_index = 45
    bottom_index = 595
    left_index = 205
    right_index = 435
    top_left_coord = (200, 40)
    bottom_right_coord = (440, 600)


else:
    categorical_model = load_model('models/MobileNetV2_shape224.h5')
    binary_models = {'ayataka' : load_model('models/ayataka.h5'),
                    'cocacola' : load_model('models/cocacola.h5'),
                    'craft_boss_black' : load_model('models/craft_boss_black.h5'),
                    'energy_peaker' : load_model('models/energy_peaker.h5'),
                    'ilohas' : load_model('models/ilohas.h5')}
    top_index = 55
    bottom_index = 695
    left_index = 555
    right_index = 795
    top_left_coord = (550, 50)
    bottom_right_coord = (800, 700)


# 画像撮影 & 判定
if __name__ == "__main__":
    while True:
        total_price = 0
        print('\n...\n')
        print('welcome!\n')
        key = input('Please press "Enter" to scan the product.\n')
        
        while True:
            frame = get_image(cap, raspi_flg) # 撮影した画像から商品部分を切り取る
            dst = frame[top_index:bottom_index, left_index:right_index]
            cv2.rectangle(frame, top_left_coord, bottom_right_coord, box_color)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # クラス分類（飲料 or その他）
            class_name = detect(rgb_frame, class_names, args.prototxt, args.caffemodel)

            # 「bottle」が含まれたクラスが検出された場合
            if bottle_str in class_name:

                cv2.putText(frame, push_text, text_coord, font, size, color, weight)
                cv2.imshow("detections", frame)

                # エンターキー押下時
                if cv2.waitKey(1) & 0xFF == 13:
                    t1 = time.time()
                    cv2.imwrite(img_path, dst)
                    drink_name, drink_price = categorical_pred(categorical_model)
                    binary_model = None if raspi_flg else binary_models[drink_name]

                    if not binary_pred(binary_model) or drink_name == 'unknown':
                        print('unknown item was placed.')
                        os.remove(img_path)
                        continue
                    
                    # 商品名、単価を出力する
                    print('{} : {} F\n'.format(drink_name, drink_price))
                    t2 = time.time()
                    elapsed_time = t2 - t1
                    print("elapsed_time：{} s\n".format(elapsed_time))
                    
                    # 合計金額をアップデート（単価を加算）
                    total_price += drink_price
                    
                    # ユーザーが商品購入の継続要否を入力
                    key = input('Press "y + Enter" to scan products continuously, or "Enter" to check\n')
                    
                    # 商品購入を継続しない場合、精算する
                    if key != 'y':
                        print("Total amount :{} F\n".format(total_price))
                        break
                    
                    # 画像を消去
                    os.remove(img_path)

            else:
                cv2.putText(frame, desc_text, text_coord, font, size, color, weight)
                cv2.imshow("detections", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
