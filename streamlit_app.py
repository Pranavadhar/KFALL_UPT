import streamlit as st
import torch
import torch.nn as nn

class KFALL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  
        return out


net = KFALL(input_dim=13, hidden_dim=512, output_dim=3, layers=5)
net.load_state_dict(torch.load("model.pth"))
net.eval()

st.title('KFALL FALL PREDICTION')

st.sidebar.header('Input Features')
t_sin_s = st.number_input('t_sin_s')
t_cos_s = st.number_input('t_cos_s')
t_sin_m = st.number_input('t_sin_m')
t_cos_m = st.number_input('t_cos_m')
AccX = st.number_input('AccX')
AccY = st.number_input('AccY')
AccZ = st.number_input('AccZ')
GyrX = st.number_input('GyrX')
GyrY = st.number_input('GyrY')
GyrZ = st.number_input('GyrZ')
EulerX = st.number_input('EulerX')
EulerY = st.number_input('EulerY')
EulerZ = st.number_input('EulerZ')

input_tensor = torch.FloatTensor([[[t_sin_s, t_cos_s, t_sin_m, t_cos_m, AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ]]])

with torch.no_grad():
    output = net(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

st.sidebar.header('Output')
st.sidebar.write(f'Predicted Class: {predicted_class}')
st.sidebar.write('Probabilities:')
st.sidebar.write(f'No Fall: {probabilities[0][0]:.4f}')
st.sidebar.write(f'Pre-Impact Fall: {probabilities[0][1]:.4f}')
st.sidebar.write(f'Fall: {probabilities[0][2]:.4f}')
