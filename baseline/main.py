import numpy as np
import torch
import torch.nn as nn
from config import *
import time
import os
from model import *
from ctcdecode import CTCBeamDecoder

if __name__ == '__main__':
    model = DeepSpeech()
    decoder = CTCBeamDecoder(['$']*output_size, beam_width=5, blank_id=0, num_processes=os.cpu_count(), log_probs_input=True)
    
    batch_size = 3
    seq_len = 9
    n_iter = 100
    
    inp = torch.ones((batch_size, seq_len, input_size+2*input_size*n_context))

    start_time = time.perf_counter()
    for i in range(n_iter):
        out = model(inp)
    end_time = time.perf_counter()
    print("Forward: %f s" % ((end_time-start_time)/n_iter))

    start_time = time.perf_counter()
    out = out.transpose(0, 1)
    out_lens = torch.tensor([seq_len for _ in range(batch_size)])
    output, scores, timesteps, out_seq_len = decoder.decode(out, out_lens) # [b, seq_len, vocab_size] -> [b, beam_width, seq_len]
    end_time = time.perf_counter()
    print("CTC Decode %f s" % (end_time-start_time))