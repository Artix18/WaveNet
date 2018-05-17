import time
import torch
import numpy as np
import ipdb


import wavenet.config as config
from wavenet.model import WaveNet
import wavenet.utils.data as utils



lang = """
def wavenet1(
    float(B, RESIDUAL_C, RECEPTIVE_FIELD) Data,
    float(DILATION_C, RESIDUAL_C, 2) FilterWeight,
    float(DILATION_C, RESIDUAL_C, 2) GateWeight,
    float(DILATION_C) FilterBias,
    float(DILATION_C) GateBias,
    float(RESIDUAL_C, DILATION_C) ResWeight,
    float(RESIDUAL_C) ResBias,
    float(SKIP_C, DILATION_C) SkipWeight,
    float(SKIP_C) SkipBias,
    float(DILATION_FACTOR) Dilation)
    -> (FilterOut, GateOut, NonLin, Res, Skip)
{
    FilterOut(b, dil, rf)   = FilterBias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    FilterOut(b, dil, rf)  += Data(b, r_res, rf) * FilterWeight(dil, r_res, 1) +
        (
          (rf - DILATION_FACTOR >= 0) ?
            Data(b, r_res, rf - DILATION_FACTOR) * FilterWeight(dil, r_res, 0) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    GateOut(b, dil, rf)   = GateBias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    GateOut(b, dil, rf)  += Data(b, r_res, rf) * GateWeight(dil, r_res, 1) +
        (
          (rf - DILATION_FACTOR >= 0) ?
            Data(b, r_res, rf - DILATION_FACTOR) * GateWeight(dil, r_res, 0) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    NonLin(b, dil, rf)   =         tanh(FilterOut(b, dil, rf))
        where rf in 0:RECEPTIVE_FIELD
    NonLin(b, dil, rf)  *= 1 / (1 + exp( -GateOut(b, dil, rf)))
        where rf in 0:RECEPTIVE_FIELD

       Res(b, res, rf)   =   Data(b,  res, rf) + ResBias(res)
       Res(b, res, rf)  += NonLin(b, r_in, rf) * ResWeight(res, r_in)

      Skip(b, skip, rf) +=! NonLin(b, r_dil, rf) * SkipWeight(skip, r_dil)
        where rf in 0:RECEPTIVE_FIELD
      Skip(b, skip, rf)  = Skip(b, skip, rf) + SkipBias(skip)
        where rf in 0:RECEPTIVE_FIELD
}
"""

layer_size = 1
stack_size = 1
in_channels = 32 #skip channel
res_channels = 32 #res channel / dil channel

wavenet = WaveNet(layer_size, stack_size, in_channels, res_channels)

B, RESIDUAL_C, RECEPTIVE_FIELD = 32, 32, 32
DILATION_C, SKIP_C, DILATION_FACTOR = 32, 32, 1

Data, FilterWeight, GateWeight, FilterBias, GateBias, ResWeight, ResBias, SkipWeight, SkipBias, Dilation = torch.randn(B, RESIDUAL_C, RECEPTIVE_FIELD).cuda(), torch.randn(DILATION_C, RESIDUAL_C, 2).cuda(), torch.randn(DILATION_C, RESIDUAL_C, 2).cuda(), torch.randn(DILATION_C).cuda(), torch.randn(DILATION_C).cuda(), torch.randn(RESIDUAL_C, DILATION_C).cuda(), torch.randn(RESIDUAL_C).cuda(), torch.randn(SKIP_C, DILATION_C).cuda(), torch.randn(SKIP_C).cuda(), torch.randn(DILATION_FACTOR).cuda()

#dilated1 = tanh = filter
FilterWeight = wavenet.net.module.res_stack.res_blocks[0].module.dilated1.weight
GateWeight = wavenet.net.module.res_stack.res_blocks[0].module.dilated2.weight
FilterBias = wavenet.net.module.res_stack.res_blocks[0].module.dilated1.bias
GateBias = wavenet.net.module.res_stack.res_blocks[0].module.dilated2.bias
ResWeight = wavenet.net.module.res_stack.res_blocks[0].module.conv_res.weight[:,:,0]
ResBias = wavenet.net.module.res_stack.res_blocks[0].module.conv_res.bias
SkipWeight = wavenet.net.module.res_stack.res_blocks[0].module.conv_skip.weight[:,:,0]
SkipBias = wavenet.net.module.res_stack.res_blocks[0].module.conv_skip.bias

#ipdb.set_trace()

import tensor_comprehensions as tc
tcwavenet = tc.define(lang, name="wavenet1")
best_options = tcwavenet.autotune(Data, FilterWeight, GateWeight, FilterBias, GateBias, ResWeight, ResBias, SkipWeight, SkipBias, Dilation)
out = tcwavenet(Data, FilterWeight, GateWeight, FilterBias, GateBias, ResWeight, ResBias, SkipWeight, SkipBias, Dilation, options=best_options)

#for name, param in wavenet.net.module.res_stack.res_blocks[0].state_dict().items():
#    print(name, param.size())

dataWavenet = Data.permute(0,2,1)
out2 = wavenet.generate(dataWavenet)

#print(out2.shape)
#for truc in out:
#    print(truc.shape)

#print(out[-1][:,:,1:])
#print(out2.permute(0,2,1))

out = out[-1][:,:,1:]
out2 = out2.permute(0,2,1)

print("erreur absolue totale :")
print(torch.abs(out - out2).sum())

print("erreur relative moyenne :")
diff = torch.abs((out - out2) / out2)
print(diff.mean())

print("Tapez C pour passer au benchmarking")
ipdb.set_trace()

warmup = 100
iters=1000
for i in range(warmup):
    tcwavenet(Data, FilterWeight, GateWeight, FilterBias, GateBias, ResWeight, ResBias, SkipWeight, SkipBias, Dilation, options=best_options)
    #tcwavenet(Data, options=best_options)
    torch.cuda.synchronize()

liste_t_tc = []
now = time.clock()
for i in range(iters):
    before = time.clock()
    tcwavenet(Data, FilterWeight, GateWeight, FilterBias, GateBias, ResWeight, ResBias, SkipWeight, SkipBias, Dilation, options=best_options)
    #tcwavenet(Data)
    torch.cuda.synchronize()
    after = time.clock()
    liste_t_tc.append(after - before)
torch.cuda.synchronize()
total_time = (time.clock() - now)
mean_time = total_time / iters

print("{0} Mean time: {1} us".format("wavenet TC", mean_time * 1e6))

liste_t_tc.sort()
p50 = liste_t_tc[iters//2] #np.mean(liste_t_tc[50:-50])
print("{0} p50 time: {1} us".format("wavenet TC", p50*1e6))

#warmup = 10
#iters=100
for i in range(warmup):
    wavenet.generate(dataWavenet)
    torch.cuda.synchronize()

liste_t_pt = []
now = time.clock()
for i in range(iters):
    before = time.clock()
    wavenet.generate(dataWavenet)
    torch.cuda.synchronize()
    after = time.clock()
    liste_t_pt.append(after - before)
torch.cuda.synchronize()
total_time = (time.clock() - now)
mean_time = total_time / iters

print("{0} Mean time: {1} us".format("wavenet pytorch", mean_time * 1e6))

liste_t_pt.sort()
p50 = liste_t_pt[iters//2] #np.mean(liste_t_pt[50:-50])
print("{0} p50 time: {1} us".format("wavenet pytorch", p50*1e6))
