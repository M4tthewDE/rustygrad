import torch

def run_batchnorm2d():
    num_features = 4
    #weight = torch.randn(num_features)
    #bias = torch.randn(num_features)
    #running_mean = torch.randn(num_features)
    #running_var = torch.randn(num_features)

    weight = torch.tensor([-0.23390397,  0.80346787, -0.8997578,  -0.72561103])
    bias = torch.tensor([ 0.05118884, -1.8861961,  -0.48187035,  0.77148885])
    running_mean = torch.tensor([-0.8936946,   0.88154835,  0.50745213, -0.63631004])
    running_var = torch.tensor([1.9885837,   1.4461043,    0.68407005, -0.32585317])

    print("WEIGHT", weight.detach().numpy())
    print("BIAS", bias.detach().numpy())
    print("RUNNING_MEAN", running_mean.detach().numpy())
    print("RUNNING_VAR", running_var.detach().numpy())

    with torch.no_grad():
        bn = torch.nn.BatchNorm2d(num_features).eval()
        bn.training = True 
        bn.weight[:] = weight
        bn.bias[:] = bias
        bn.running_mean[:] = running_mean
        bn.running_var[:] = running_var

    #inn = torch.rand([2, num_features, 3, 3])
    inn = torch.tensor([[[[7.2550803e-01, 2.6468736e-01, 1.6042733e-01],
       [9.1248131e-01, 1.6140634e-01, 2.5769627e-01],
       [4.6988237e-01, 4.1404659e-01, 4.2651403e-01]],

      [[6.2887025e-01, 2.9633635e-01, 8.1406182e-01],
       [3.3209044e-01, 2.3387557e-01, 7.6986372e-02],
       [7.9895943e-01, 4.6219945e-01, 5.4436874e-01]],

      [[5.0352424e-01, 7.4825037e-01, 9.6039361e-01],
       [4.1384923e-01, 6.5281290e-01, 4.5812815e-01],
       [4.9625474e-01, 8.5442269e-01, 2.3508650e-01]],

      [[8.1202835e-01, 5.9161729e-01, 8.7563610e-01],
       [7.8307271e-01, 2.7927721e-01, 5.9114403e-01],
       [5.6509542e-01, 6.7465717e-01, 5.4024965e-01]]],


     [[[6.9285572e-01, 7.9642236e-01, 3.3424616e-02],
       [3.6928445e-01, 3.0962038e-01, 5.8036125e-01],
       [6.5488166e-01, 2.0787734e-01, 3.7388825e-01]],

      [[4.8392904e-01, 8.8160717e-01, 1.0005343e-01],
       [6.3134187e-01, 9.7084045e-04, 9.5077854e-01],
       [5.2091897e-02, 8.1263256e-01, 2.0272434e-02]],

      [[9.7554046e-01, 4.3832892e-01, 3.4956914e-01],
       [5.9617567e-01, 1.4704645e-02, 5.1316148e-01],
       [5.0387317e-01, 4.3340671e-01, 5.0901908e-01]],

      [[9.7953200e-02, 3.7877357e-01, 3.9520144e-01],
       [1.4133233e-01, 1.1132270e-01, 9.3504202e-01],
       [8.0020428e-01, 4.0364802e-01, 2.0521462e-02]]]])
    print("INPUT")
    print(inn.detach().numpy())

    out = bn(inn)
    print("OUTPUT")
    print(out.shape)
    print(out.detach().numpy())


if __name__ == "__main__":
    run_batchnorm2d()
