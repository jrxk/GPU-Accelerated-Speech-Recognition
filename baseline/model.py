import numpy as np
import torch
import torch.nn as nn
from config import *

class DeepSpeech(nn.Module):
    def __init__(self):
        super(DeepSpeech, self).__init__()
        self.mlp123 = nn.Sequential(
            nn.Linear(input_size+2*input_size*n_context, hidden_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.RNN(hidden_3, rnn_hidden_size, num_layers=1, bidirectional=False) # bidir true
        self.mlp56 = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_5, output_size),
        )
    
    def forward(self, x):
        # x: [b, seq_len, n_input + 2*input_size*n_context]
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.permute(1, 0, 2)
        x = x.reshape(seq_len*batch_size, -1)
        x = self.mlp123(x)
        x = x.reshape(seq_len, batch_size, hidden_3)
        x, _ = self.rnn(x)
        x = x.reshape(seq_len*batch_size, rnn_hidden_size)
        x = self.mlp56(x)
        x = x.reshape(seq_len, batch_size, output_size)
        return x.log_softmax(2)