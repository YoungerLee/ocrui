import collections
import numpy as np
alphabet ='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'
text_list = ['03JiuWuHi-Tech409132056', '82JiuWuHi-Tech011163106',
             '72JiuWuHi-Tech609143128', '62JiuWuHi-Tech209141048',
             'x2JiuWuHi-Tech105246174', '02JiuWuHi-Tech410216126',
             '42JiuWuHi-Tech004102008', '72JiuWuHi-Tech109278305',
             '82JiuWuHi-Tech103167035', '12JiuWuHi-Tech210287068',
             '52JiuWuHi-Tech901267004', '32JiuWuHi-Tech801188235',
             '22JiuWuHi-Tech501315180', '32JiuWuHi-Tech702131126',
             '92JiuWuHi-Tech012097029', '12JiuWuHi-Tech312314066',
             'x2JiuWuHi-Tech806244170', '12JiuWuHi-Tech905066201',
             '72JiuWuHi-Tech606274181', '12JiuWuHi-Tech802133050']

# text_list = ['409132056', '011163106',
#              '609143128', '209141048',
#              '105246174', '410216126',
#              '004102008', '109278305',
#              '103167035', '210287068',
#              '901267004', '801188235',
#              '501315180', '702131126',
#              '012097029', '312314066',
#              '806244170', '905066201',
#              '606274181', '802133050']
class LabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.__alphabet = alphabet + '#'  # for `-1` index
        self._dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self._dict[char] = i
            
    def encode_sparse_tensor(self, text_list):
        """ Create a sparse representention of x.
        Args: sequences: a list of lists of type dtype where each element is a sequence
        Returns: A tuple with (indices, values, shape)
        """
        sequences, length = self.__encode_sequence(text_list)
        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
        return (indices, values, shape), np.asarray(length)

    def __encode_sequence(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): convert to sequences.

        Returns:
            sequences: list of sequence.
            length: list of length of sequence.
        """
        if isinstance(text, str):
            seq = [
                [
                    self._dict[char.lower() if self._ignore_case else char]
                    for char in text
                 ]
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            seq = [
                [
                    self._dict[char.lower() if self._ignore_case else char]
                    for char in t
                 ] for t in text
            ]
        return (seq, length)

    def decode_sparse_tensor(self, sparse_tensor, length, raw=False):
        sequences = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                sequences.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(sparse_tensor[1][offset])
        sequences.append(current_seq)
        result = self.decode_sequence(sequences, length, raw)
        return result

    def decode_sequence(self, sequences, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        blank_index = len(self.__alphabet) - 1
        if len(length) == 1:
            length = length[0]
            sequence = sequences[0]
            assert len(sequence) == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.__alphabet[i] for i in sequence])
            else:
                char_list = []
                for i in range(length):
                    if sequence[i] != blank_index and \
                            (not (i > 0 and sequence[i - 1] == sequence[i])):
                        char_list.append(self.__alphabet[sequence[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert sum([int(len(seq)) for seq in sequences]) == sum(length), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(len(length)):
                l = length[i]
                texts.append(
                    self.__decode_sequence(
                        [sequences[i]], [l], raw=raw))
                index += l
            return texts

if __name__ == '__main__':
    converter = LabelConverter(alphabet)
    sparse, length = converter.encode_sparse_tensor(['72JiuWuHi-Tech606274181'])
    decoded = converter.decode_sparse_tensor(sparse, length, raw=False)

