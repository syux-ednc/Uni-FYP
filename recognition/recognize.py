import torch
from collections import OrderedDict
from torch.autograd import Variable
from models.crnn import CRNN
from utils import strLabelConverter
from dataset import resizeNormalize
from PIL import Image

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
converter = strLabelConverter(alphabet)

class TextRecognizer:
    def __init__(self, model_path):
        print('Loading pre-trained CRNN Recognizer model')
        self.model = CRNN(32, 1, 37, 256)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            state_dict_rename[name] = v
        self.model.load_state_dict(state_dict_rename)

    def predict(self, image):
        transformer = resizeNormalize((100, 32))
        image = Image.fromarray(image).convert('L')
        image = transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()

        image = image.view(1, *image.size())
        image = Variable(image)

        self.model.eval()
        preds = self.model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        return raw_pred, sim_pred
