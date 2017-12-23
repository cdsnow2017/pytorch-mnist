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
    model = torch.load('net.pkl')
    data=np.ones((1,1,28,28))
    for i in range(28):
        for j in range(28):
            data[0][0][i][j]=sample[i][j]
    data=torch.from_numpy(data).double()
    data = Variable(data)
    model.double()
    output = model(data)
    pred = output.data.max(1)[1]
    print(pred)
    #output.data.max(0)输出10个概率值
    #output.data.max(1)输出10个(概率值,对应数值)
    #print(output.data.max(1)[1])
    output_list=output.data.numpy().tolist()
    print(output_list)
    minValue=min(output_list[0])
    maxValue=max(output_list[0])
    sumValue=sum(output_list[0])
    probability_list=[(item-minValue)/(-1.0*sumValue) for item in output_list[0]]
    return probability_list   #output_list[0]



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