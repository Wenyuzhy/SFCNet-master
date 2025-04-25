from thop import profile
import torch
import time
# from net.CIDNet import CIDNet
from net.network import SFCFormer


# model = CIDNet().to('cuda')  
model = SFCFormer().to('cuda')
input = torch.rand(1,3,256,256).to('cuda')  
torch.cuda.synchronize()
model.eval()
time_start = time.time()
_ = model(input)
time_end = time.time()
torch.cuda.synchronize()
time_sum = time_end - time_start
print(f"Time: {time_sum}")
n_param = sum([p.nelement() for p in model.parameters()])  
n_paras = f"n_paras: {(n_param/2**20)}M\n"
print(n_paras)
macs, params = profile(model, inputs=(input,)) 
print(f'Parameters:{params/(2**20)}M, FLOPs:{macs/(2**30)}G')
