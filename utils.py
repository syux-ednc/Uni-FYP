import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import cv2
import imutils

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict.get(char.lower() if self._ignore_case else char)
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


def display_image(img, width):
    img = imutils.resize(img, width=width)
    cv2.imshow('Image', img)
    print('Press "q" to quit, any other key to view next image.')
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


def format_sentence(words, debug=False):
    formatted_words = list(map(format_word, words.items()))
    # print("Formatted Words: ")
    # print(formatted_words)
    res = sentence_formatter_v1(formatted_words, debug)

    if debug:
        print(formatted_words)
        print(res)

    sentence_prettifier_v1(res)


def find_y_intercept(p1, p2):
    # Derive a y = mx + c, based on 2 points
    # p1: Bottom left point of bounding box
    # p2: Bottom right point of bounding box
    # First convert y to negative due to OpenCV2's xy direction

    p1 = [p1[0], -p1[1]]
    p2 = [p2[0], -p2[1]]
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])

    return p1[1] - (m * p1[0])

def format_word(word):
    # Will need y-intercept and x-value of bottom left corner of
    # word to begin grouping words
    return {
      'y_intercept': find_y_intercept(word[1]['vertices'][0], word[1]['vertices'][3]),
      'x_value_left': word[1]['vertices'][0][0],
      'x_value_right': word[1]['vertices'][3][0],
      'word': word[1]['pred_text']
    }

y_threshold = 30
x_threshold = 60

def sentence_formatter_v1(words, debug=False):
    word_groups = {}

    # First sort the words based on x-value
    sorted_words = sorted(words, key=lambda x: x['x_value_left'])
    row_num = 0

    for word in sorted_words:
        if len(word_groups) == 0:
            if debug:
                print("first word is", word['word'])

            word_groups[f'row_{row_num}'] = [word]
            row_num += 1

        else:
            for row in list(word_groups):
                if (abs(word_groups[row][len(word_groups[row]) - 1]['y_intercept'] - word['y_intercept']) < y_threshold):

                    if debug:
                        print(word['word'], word['y_intercept'], "in the same row as", word_groups[row][len(word_groups[row]) - 1]['word'], word_groups[row][len(word_groups[row]) - 1]['y_intercept'])

                    word_groups[row].append(word)
                    break

                else:
                    if int(row.split('_')[1]) == row_num - 1:

                        if debug:
                            print(word['word'], "in a new row")

                        word_groups[f'row_{row_num}'] = [word]
                        row_num += 1

    # Then we sort the rows based on y-intercept
    word_groups = {k:v for k, v in sorted(word_groups.items(), key = lambda x: x[1][0]['y_intercept'], reverse=True)}

    # Now we have to detect which words are clusters and separate accordingly
    # Will need to think of a better data structure to have cluster context
    # Now is just treating it as a new row
    for key in list(word_groups):
        for idx, word in enumerate(word_groups[key]):
            if (idx != len(word_groups[key]) - 1 and
                word['x_value_right'] < word_groups[key][idx + 1]['x_value_left'] and
                abs(word['x_value_right'] - word_groups[key][idx + 1]['x_value_left']) > x_threshold):
                print('different cluster')
                word_groups[f'row_{row_num}'] = [word]
                row_num += 1
                word_groups[key].pop(idx)


    # Finally, simplify the data structure
    res = []
    for item in word_groups.items():
        res.append([x['word'] for x in item[1]])

    return res


def sentence_prettifier_v1(words):
    print('========================================== \n'
          '| Detected Texts                         | \n'
          '==========================================')
    for idx, word in enumerate([' '.join(items) for items in words]):
        print(f'| Row {idx+1} | {word:30} |')
    print('==========================================')

