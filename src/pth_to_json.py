import json
import torch
import sys
from json import JSONEncoder
from torch.utils.data import Dataset

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(JSONEncoder, self)

def main(file_path):
    model = torch.load(file_path)
    dump = json.dumps(model, cls=EncodeTensor)
    with open(file_path + '.json', 'w') as outfile:
        outfile.write(dump)

if __name__ == '__main__':
    main(sys.argv[1])
