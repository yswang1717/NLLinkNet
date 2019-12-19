from networks.dinknet import LinkNet34, DinkNet34
from networks.nllinknet_location import NL3_LinkNet, NL4_LinkNet, NL34_LinkNet, Baseline


if __name__ == '__main__':
    import torch
    from thop import profile

    # model = LinkNet34() # LinkNet: 109.604397056, 21.6424
    # model = DinkNet34() # 134.451429376, 31.096128
    # model = NL3_LinkNet() # 125.336158208, 21.69024
    # model = NL4_LinkNet() # 125.329342464, 21.78912
    # model = NL34_LinkNet() # 125.879844864, 21.822464
    # model = Baseline() # 124.785655808, 21.656896

    input = torch.randn(1, 3, 1024, 1024)
    flops, params = profile(model, inputs=(input,))
    print("LinkNet: %s, %s" % (flops/1e9, params/1e6))

