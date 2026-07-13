from .custom_dataset import Imagenette2Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ["get_loader"]

def get_loader(config, mode='train'):
    assert mode in ["train", "val", "inference"], f"인자가 잘못되었습니다! 입력된 mode: '{mode}' ('train', 'val', 'inference' 중 하나여야 합니다)"


    train_transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["data"]["mean"], std=config["data"]["std"])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["data"]["mean"], std=config["data"]["std"])
    ])

    transform = train_transform if mode == "train" else val_transform
    is_train = (mode == "train")

    dataset = Imagenette2Dataset(config["data"][mode + "_dir"], transform)

    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=is_train, drop_last=is_train, num_workers=config["data"]["num_workers"], pin_memory=True)

    return loader