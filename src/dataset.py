import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DreamyDataset(Dataset):
    """
    Dataset cho ảnh dreamy landscape đã được resize sẵn.
    Trả về Tensor với giá trị đã normalize về [-1, +1].
    """
    def __init__(self, root_dir: str, size: int = 512):
        """
        Args:
            root_dir: thư mục chứa ảnh đã preprocess (data/processed)
            size: kích thước ảnh (chắc chắn là vuông size x size)
        """
        self.root_dir = root_dir
        self.paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),                                # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],                # (x - 0.5)/0.5
                                 [0.5, 0.5, 0.5])                # => [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
