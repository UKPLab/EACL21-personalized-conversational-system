def get_ngrams(text, n):
    """Returns all ngrams that are in the text.
    Inputs:
        text: string
        n: int
    Returns:
        list of strings (each is a ngram)
    """
    if text == "":
        return []
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - (n - 1))]  # list of str


def intrep_frac(lst):
    """Returns the fraction of items in the list that are repeated"""
    if len(lst) == 0:
        return 0
    num_rep = 0
    for idx in range(len(lst)):
        if lst[idx] in lst[:idx]:
            num_rep += 1
    return num_rep / len(lst)


def flatten(list_of_lists):
    """Flatten a list of lists"""
    return [item for sublist in list_of_lists for item in sublist]


def extrep_frac(lst1, lst2):
    """Returns the fraction of items in lst1 that are in lst2"""
    if len(lst1) == 0:
        return 0
    num_rep = len([x for x in lst1 if x in lst2])
    return num_rep / len(lst1)
