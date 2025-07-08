from torch.utils.data import DataLoader
from dataset import DreamyDataset
import matplotlib.pyplot as plt
import torchvision.utils as vutils

if __name__ == "__main__":
    dataset = DreamyDataset("data/processed", size=512)
    loader  = DataLoader(dataset,
                         batch_size=4,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=True)
    batch = next(iter(loader))  # lấy batch đầu tiên
    print("Batch shape:", batch.shape)

    grid = vutils.make_grid(batch[:4], nrow=4, normalize=True, scale_each=True)
    plt.figure(figsize=(8,4))
    plt.axis('off')
    plt.title('Sample batch')
    plt.imshow(grid.permute(1,2,0))
    plt.show()
