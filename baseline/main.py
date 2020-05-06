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
    vocab_size = config["vocab_size"]
    # num_processes = config["num_processes"]
    
    beam_width = config["beam_width"]
    # print("num_processes_cpu: ", os.cpu_count())

    if device == "cpu":
        num_threads = config["num_threads"]
        torch.set_num_threads(num_threads)
        print("num_threads: " ,torch.get_num_threads())
    model = DeepSpeech(config)
    decoder = CTCBeamDecoder(['$']*(vocab_size + 1), beam_width=beam_width, blank_id=0, num_processes=num_threads, log_probs_input=True)

    # inp = torch.ones((batch_size, seq_len, input_size+2*input_size*n_context))
    
    model = model.to(device)
    
    forward_time = 0
    decode_time = 0
    overall_time = 0
    for i in range(n_iter):
        start_time = time.perf_counter()
        inp = torch.rand((batch_size, seq_len, input_size+2*input_size*n_context))
        inp = inp.to(device)
        out = model(inp)
        end_time1 = time.perf_counter()
        start_time1 = time.perf_counter()
        out = out.transpose(0, 1)
        out_lens = torch.tensor([seq_len for _ in range(batch_size)])
        output, scores, timesteps, out_seq_len = decoder.decode(out, out_lens) # [b, seq_len, vocab_size] -> [b, beam_width, seq_len]
        end_time2 = time.perf_counter()

        forward_time += end_time1 - start_time
        decode_time += end_time2 - start_time1
        overall_time += end_time2 - start_time
    

    print("Forward: %f s" % (forward_time/n_iter))
    print("CTC Decode %f s" % (decode_time/n_iter))
    print("Overall %f s" % (overall_time/n_iter))


if __name__ == '__main__':
    configs = json.load(open(sys.argv[1]))
    for config in configs:
        print("====== config ======")
        print(config)
        print("====================")
        run(config)



