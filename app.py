from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd

app = Flask(__name__)

ref = pd.read_csv('plants2.csv', header=None, index_col=0, squeeze=True).to_dict()
model = load_model("best_model.h5")
model.make_predict_function()
def predict(path):
  img = load_img(path, target_size=(256,256))

  i = img_to_array(img)

  im = preprocess_input(i)

  img = np.expand_dims(im, axis=0)
  
  pred = np.argmax(model.predict(img))

  return ref[pred]

@app.route("/")
def home():
  return render_template("index.html")
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
  if request.method == 'POST':
    img = request.files["fileName"]
    
    path = "static/" + img.filename
    img.save(path)
    
    p = predict(path)
   return render_template("index.html", prediction =p, path=path)
