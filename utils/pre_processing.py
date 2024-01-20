# sequence_length is directly propotional to BLUE score
class SentenceFilter:
    def __init__(self, x: list[str], y:list[str], min_length: int, max_length: int) -> None:
        self.x = x
        self.y = y
        self.min_length = min_length
        self.max_length = max_length
        self.unique_sentences = set()

    def is_valid_length(self, sentence: str) -> bool:
        return self.min_length <= len(sentence) <= self.max_length

    def __iter__(self):
        for xi, yi in (zip(self.x, self.y)):
            if self.is_valid_length(xi) and self.is_valid_length(yi):
                sentence_pair = (xi, yi)
                yield sentence_pair



def pop_sample(x: list[str], y: list[str], min_length: int, max_length: int) -> tuple[list[str],list[str]]:
    sentence_filter = SentenceFilter(x, y, min_length, max_length)
    new_x, new_y = zip(*sentence_filter)
    return list(new_x), list(new_y)

def load_data(x_path,y_path,min_len: int = 5,max_len: int = 254) -> tuple[list[str], list[str]]:
  """
    Load data from specified paths and filter sentences based on length constraints.

    Parameters:
    - x_path (str): Path to the source data file.
    - y_path (str): Path to the target data file.
    - min_len (int): Minimum allowed length for sentences (default: 0).
    - max_len (int): Maximum allowed length for sentences (default: 254).

    Returns:
    tuple[list[str], list[str]]: Tuple containing lists of source and target sentences.
  """
  x = pickle(x_path)
  y = pickle(y_path)
  x,y = pop_sample(x,y,min_len,max_len)
  return x,y

