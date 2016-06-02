from collections import Counter
import re


class ConllEntry:
    def __init__(self, id, form, pos, cpos, parent_id=None, relation=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.pred_parent_id = None
        self.pred_relation = None


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence])
            posCount.update([node.pos for node in sentence])
            relCount.update([node.relation for node in sentence])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), relCount.keys())


def read_conll(fh):
    root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', -1, 'rroot')
    tokens = [root]
    for line in fh:
        tok = line.strip().split()
        if not tok:
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            tokens.append(ConllEntry(int(tok[0]), tok[1], tok[4], tok[3], int(tok[6]) if tok[6] != '_' else -1, tok[7]))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write('\t'.join([str(entry.id), entry.form, '_', entry.cpos, entry.pos, '_', str(entry.pred_parent_id), entry.pred_relation, '_', '_']))
                fh.write('\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

