import streamlit as st
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model


### 画像ローダー
uploaded_file = st.file_uploader("判定したい画像をアップロードしてください。",type = ['png','jpg','jpeg'])

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("src/keras_model.h5", compile=False)

# Load the labels
class_names = open("src/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 画像判定関数
def predict(uploaded_file):

  # 画像を分析できる形に変換するため、アップロードファイルを「RGB」に変換
  image = Image.open(uploaded_file).convert('RGB')

  # 画像を224x224にリサイズする。
  size = (224, 224)
  image = ImageOps.fit(image, size)

  # 画像分析できるようにNumpy配列に変換(形状：224,224,3)
  image_array = np.asarray(image)

  # データのスケールを扱いやすく整えるために正規化（最小値0、最大値1)
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
  
  # 配列に格納
  data[0] = normalized_image_array

  # 画像を判定
  prediction = model.predict(data) #推論を実行する(分類結果の当てはまりをクラス毎に0〜1で出力)
  index = np.argmax(prediction) #結果のうち最大値のインデックスを取得
  class_name = class_names[index] #インデックスに対応するラベル(0 cat or 1 dog)を取得
  confidence_score = round((prediction[0][index]*100),2) #信頼度スコアを取得し、100倍し、小数２位まで表示
 
  # 判定結果を格納
  result_name = class_name[2:]  #ラベルから「golf or not」を取得

  # 結果を返す

  return [result_name, str(confidence_score)]

# 画像がアップロードされているかを確認
if uploaded_file is not None:
    try:
        # 画像を判定
        pred = predict(uploaded_file)
        # 情報を表示（画像判定で取得した名前と信頼度を表示する）
        st.info(f"この画像は「{pred[0]}」です(信頼度：{pred[1]}%)。")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.warning("画像がアップロードされていません。")