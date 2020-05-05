import numpy as np
import torch
import torch.nn as nn
from config import *
import time
import json
import os
import sys
from model import *
from ctcdecode import CTCBeamDecoder

def run(config):
    batch_size = config["batch_size"]
    seq_len = config["seg_len"]
    n_iter = config["epoch"]
    input_size = config["input_size"]
    device = config["device"]
    num_processes = config["num_processes"]
    # print("num_processes_cpu: ", os.cpu_count())
    model = DeepSpeech(config)
    decoder = CTCBeamDecoder(['$']*output_size, beam_width=5, blank_id=0, num_processes=num_processes, log_probs_input=True)

    # inp = torch.ones((batch_size, seq_len, input_size+2*input_size*n_context))
    inp = torch.rand((batch_size, seq_len, input_size+2*input_size*n_context))
    print("inp shape:", inp.shape)
    model = model.to(device)
    inp = inp.to(device)


    start_time = time.perf_counter()
    for i in range(n_iter):
        out = model(inp)
    end_time = time.perf_counter()
    print("Forward: %f s" % ((end_time-start_time)/n_iter))

    start_time1 = time.perf_counter()
    out = out.transpose(0, 1)
    out_lens = torch.tensor([seq_len for _ in range(batch_size)])
    output, scores, timesteps, out_seq_len = decoder.decode(out, out_lens) # [b, seq_len, vocab_size] -> [b, beam_width, seq_len]
    end_time = time.perf_counter()
    print("CTC Decode %f s" % (end_time-start_time1))
    print("Overall %f s" % (end_time-start_times))


if __name__ == '__main__':
    configs = json.load(open(sys.argv[1]))
    for config in configs:
        print("====== config ======")
        print(config)
        print("====================")
        run(config)



