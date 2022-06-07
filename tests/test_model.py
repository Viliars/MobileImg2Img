import torch
from models.network_splitsr import SplitSR

def main():
    upscale = 4
    window_size = 8
    height = 1920
    width = 1080
    model = SplitSR()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = torch.randn((8, 3, 128, 128))


    x = x.to(device) 
    x = model(x)
    print(x.shape)

if __name__ == '__main__':
    main()