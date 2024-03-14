import torch 
class label2prompt():
    def __init__(self):
        self.label2img_dict = {0:'A whole fish, tench, Tinca tinca.',
                  1:'A whole fish, goldfish, Carassius auratus.',
                  2:'A whole fish, great white shark, white shark, Carcharodon carcharias.',
                  3:'A whole fish, tiger shark.',
                  4:'A whole fish, hammerhead shark.',
                  5:'A whole fish, torpedo, electric ray, crampfish, numbfish.',
                  6:'A whole bird, brambling, Fringilla montifringilla.',
                  7:'A whole bird, goldfinch, Carduelis carduelis.',
                  8:'A whole bird, house finch, linnet, Carpodacus mexicanus.',
                  9:'A whole bird, junco, snowbird.',
                  10:'A whole bird, indigo bunting, indigo finch, indigo bird, Passerina cyanea.',
                  11:'A whole bird, robin, American robin, Turdus migratorius.',
                  12:'A whole snake, thunder snake, worm snake, Carphophis amoenus.',
                  13:'A whole snake, ringneck snake, ring-necked snake, ring snake.',
                  14:'A whole snake, hognose snake, puff adder, sand viper.',
                  15:'A whole snake, green snake, grass snake.',
                  16:'A whole snake, king snake, kingsnake.',
                  17:'A whole snake, garter snake, grass snake.',
                  18:'A whole dog, Pekinese, Pekingese, Peke.',
                  19:'A whole dog, Shih-Tzu.',
                  20:'A whole dog, Blenheim spaniel.',
                  21:'A whole dog, papillon.',
                  22:'A whole dog, Rhodesian ridgeback.',
                  23:'A whole dog, bloodhound, sleuthhound.',
                  24:'A whole butterfly, admiral.',
                  25:'A whole butterfly, ringlet butterfly.',
                  26:'A whole butterfly, monarch, monarch butterfly, milkweed butterfly, Danaus plexippus.',
                  27:'A whole butterfly, cabbage butterfly.',
                  28:'A whole butterfly, sulphur butterfly, sulfur butterfly.',
                  29:'A whole butterfly, lycaenid.',
                  30:'A whole insect, fly.',
                  31:'A whole insect, bee.',
                  32:'A whole insect, grasshopper, hopper.',
                  33:'A whole insect, cockroach, roach.',
                  34:'A whole insect, mantis, mantid.',
                  35:'A whole insect, cicada, cicala.'
                  }
    def map_labels_to_prompts(self, label_tensor):
        prompt_list = []
        labels = label_tensor.reshape(-1)
        bs = labels.shape[0]
        for index in range(bs):
            prompt_list.append(self.label2img_dict[labels[index].item()])

        return prompt_list

if __name__ == "__main__":
    toprompt = label2prompt()
    targets = torch.randint(low=0, high=6, size = [8])
    prompts = toprompt.map_labels_to_prompts(targets)
    print(prompts)