from .seq_imagenet_r import MyImageNetRDataset
import torch

class LabelConverter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        if self.dataset_name == 'seq-cifar10':
            self.labels = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        elif self.dataset_name == 'seq-cifar100':
            self.labels = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
                'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                'bowl', 'boy', 'bridge', 'bus', 'butterfly',
                'camel', 'can', 'castle', 'caterpillar', 'cattle',
                'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                'plain', 'plate', 'poppy', 'porcupine', 'possum',
                'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew',
                'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor',
                'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
            
        elif self.dataset_name == 'seq-imagenet-r':
            imagenetr_dataset = MyImageNetRDataset()
            self.labels = imagenetr_dataset.label2name
            self.name2labels = {v : k for k, v in self.labels.items()}
        else:
            raise ValueError("Unsupported dataset.")

    def convert(self, label_tensor):
        if self.dataset_name in ['seq-cifar10', 'seq-cifar100']:
            # if label_tensor.shape[0] == 1:
            #     # return self.labels[label_tensor.item()]
            #     return [self.labels[label.item()]]
            # else:
            #     return [self.labels[label.item()] for label in label_tensor]
            np_array = label_tensor.cpu().numpy()
            names = []
            for i in range(np_array.shape[0]):
                for j in range(np_array.shape[1]):
                    names.append(self.labels[np_array[i, j]])
        else:
            np_array = label_tensor.cpu().numpy()
            names = []
            for i in range(np_array.shape[0]):
                for j in range(np_array.shape[1]):
                    names.append(self.labels[str(np_array[i, j])])
        return names
            # if label_tensor.shape[0] == 1:
            #     return self.labels[str(label_tensor.item())]
            # else:
            #     return [self.labels[str(label.item())] for label in label_tensor]
            
    def name2label(self, name):
        if self.dataset_name in ['seq-cifar10', 'seq-cifar100']:
            return self.labels.index(name)
        else:
            return int(self.name2labels[name])
        
        

if __name__ =="__main__":
    convert = LabelConverter('seq-cifar10')
    pred = torch.tensor([[1, 0]], device='cuda:0')
    name_list_topk = convert.convert(pred)
    print(name_list_topk)