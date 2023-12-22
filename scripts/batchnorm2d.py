import torch

def run_batchnorm2d():
    num_features = 4
    weight = torch.randn(num_features)
    bias = torch.randn(num_features)
    running_mean = torch.randn(num_features)
    running_var = torch.randn(num_features)

    print("WEIGHT", weight.detach().numpy())
    print("BIAS", bias.detach().numpy())
    print("RUNNING_MEAN", running_mean.detach().numpy())
    print("RUNNING_VAR", running_var.detach().numpy())

    with torch.no_grad():
        bn = torch.nn.BatchNorm2d(num_features).eval()
        bn.training = False
        bn.weight[:] = weight
        bn.bias[:] = bias
        bn.running_mean[:] = running_mean
        bn.running_var[:] = running_var

    inn = torch.rand([2, num_features, 3, 3])
    print("INPUT")
    print(inn.detach().numpy())

    out = bn(inn)
    print("OUTPUT")
    print(out.detach().numpy())


if __name__ == "__main__":
    run_batchnorm2d()
