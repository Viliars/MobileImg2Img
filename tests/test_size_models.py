from prettytable import PrettyTable
from ptflops import get_model_complexity_info
from models.network_unet import UNet


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    #print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def model_complexity(model):
    model.requires_grad_(True)
    size = (512, 512)
    macs, total_s = get_model_complexity_info(model, (3, *size), as_strings=False,
                                                    print_per_layer_stat=False, verbose=False)
    gmacs = macs / (10**9)
    print(size, "=>", gmacs)


    size = (1024, 1024)
    macs, total_s = get_model_complexity_info(model, (3, *size), as_strings=False,
                                                    print_per_layer_stat=False, verbose=False)
    gmacs = macs / (10**9)
    print(size, "=>", gmacs)


def main():
    model = UNet(3, 3, 32)
    model = model.cuda()

    count_parameters(model)
    model_complexity(model)

if __name__ == '__main__':
    main()