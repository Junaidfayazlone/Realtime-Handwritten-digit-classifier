
##Importing neccassary Libraries
import torch
import torchvision
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F
input_size=784   #28X28 pixel of image
hidden_size1=200 #size of 1st hidden layer(number of perceptron)
hidden_size2=150 #size of second hidden layer
hidden_size3=100 #size of third hidden layer
hidden_size=80   #size of fourth hidden layer
output =10       #output layer
class MNIST(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2
                       ,hidden_size3,hidden_size,output):
        super(MNIST,self).__init__()
        self.f_connected1=nn.Linear(input_size,hidden_size1)
        self.f_connected2=nn.Linear(hidden_size1,hidden_size2)
        self.f_connected3=nn.Linear(hidden_size2,hidden_size3)
        self.f_connected4=nn.Linear(hidden_size3,hidden_size)
        self.out_connected=nn.Linear(hidden_size,output)
    def forward(self,x):
        out=F.relu(self.f_connected1(x))
        out=F.relu(self.f_connected2(out))
        out=F.relu(self.f_connected3(out))
        out=F.relu(self.f_connected4(out))
        out=self.out_connected(out)
        return out
Mnist_model=MNIST(input_size,hidden_size1,hidden_size2
                       ,hidden_size3,hidden_size,output)
Mnist_model.load_state_dict(torch.load("model_weights.pth"))
Mnist_model.eval()


from flask import Flask, render_template, request, jsonify
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home():
    return render_template('index.html') 

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json.get('image')
        base64_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        image_stream = BytesIO(image_bytes)
        prediction = predict_number(image_stream)
        print(prediction)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def predict_number(image_data):
    from PIL import Image
    original_image = Image.open(image_data)
    resized_image = original_image.resize((28, 28), Image.Resampling.LANCZOS)
    resized_image.save('resized_image.png')
    convert_tensor = transforms.ToTensor()
    input_data=convert_tensor(resized_image)
    input_data=input_data.reshape(-1,784)
    output=Mnist_model(input_data)
    _,prediction=torch.max(output,1)
    number_prediction=torch.mode(prediction,0)

    return int(number_prediction.values.data)

if __name__ == '__main__':
    app.run(debug=True)


