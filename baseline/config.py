# config
device='cpu'

n_context = 1
input_size = 10 # 26
linear_size = 40
rnn_hidden_size = 50
vocab_size = 26

hidden_1 = linear_size # 2048
hidden_2 = linear_size
hidden_5 = linear_size
hidden_3 = rnn_hidden_size
output_size = vocab_size + 1