import torch
import torch.nn as nn

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = None

    fused_conv = nn.Conv2d(conv.in_channels,
                               conv.out_channels,
                               conv.kernel_size,
                               conv.stride,
                               conv.padding,
                               groups=conv.groups,
                               bias=True)

    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


def test_net(m, input):
    import time
    s = time.time()
    o_output = m(input)
    #o_output = o_output[0]
    print("Original time: ", time.time() - s)
    fuse_module(m)

    s = time.time()
    f_output = m(input)
    #f_output = f_output[0]
    print("Fused time: ", time.time() - s)
    print(m)
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    assert (o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())
