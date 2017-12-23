import numpy
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from torch.autograd import Variable
#from model import LeNet5

# webapp
app = Flask(__name__)

def predict_with_pretrain_model(sample):
    '''
    Args:
        sample: A integer ndarray indicating an image, whose shape is (28,28).

    Returns:
        A list consists of 10 double numbers, which denotes the probabilities of numbers(from 0 to 9).
        like [0.1,0.1,0.2,0.05,0.05,0.1,0.1,0.1,0.1,0.1].
    '''
    sample = sample / 255.0
    images = np.array([])
    images = np.append(images, sample)
    img = images.reshape(-1, 1, 28, 28)
    img = torch.from_numpy(img).float()
    model = torch.load('model.pkl')
    outputs = model(Variable(img))
    _, predicted = torch.max(outputs.data, 1)
    out = F.softmax(outputs)
    p = out.data.numpy()
    p = p.reshape(-1)
    p = p.tolist()
    return p



@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(28, 28)
    output = predict_with_pretrain_model(input)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')



if __name__ == '__main__':
    app.run()