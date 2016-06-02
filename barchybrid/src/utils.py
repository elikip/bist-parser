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


class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = 0 # None
            root.pred_relation = 'rroot' # None
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)


    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1

def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP, True):
            wordsCount.update([node.norm for node in sentence])
            posCount.update([node.pos for node in sentence])
            relCount.update([node.relation for node in sentence])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())},  posCount.keys(), relCount.keys())

def read_conll(fh, proj):
    dropped = 0
    read = 0
    root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', 0, 'rroot')
    tokens = [root]
    for line in fh:
        tok = line.strip().split()
        if not tok:
            if len(tokens)>1:
                if not proj or isProj(tokens):
                    yield tokens
                else:
                    print 'Non-projective sentence dropped'
                    dropped += 1
                read += 1
            tokens = [root]
            id = 0
        else:
            tokens.append(ConllEntry(int(tok[0]), tok[1], tok[4], tok[3], int(tok[6]) if tok[6] != '_' else -1, tok[7]))
    if len(tokens) > 1:
        yield tokens

    print dropped, 'dropped non-projective sentences.'
    print read, 'sentences read.'


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

cposTable = {"PRP$": "PRON", "VBG": "VERB", "VBD": "VERB", "VBN": "VERB", ",": ".", "''": ".", "VBP": "VERB", "WDT": "DET", "JJ": "ADJ", "WP": "PRON", "VBZ": "VERB", 
             "DT": "DET", "#": ".", "RP": "PRT", "$": ".", "NN": "NOUN", ")": ".", "(": ".", "FW": "X", "POS": "PRT", ".": ".", "TO": "PRT", "PRP": "PRON", "RB": "ADV", 
             ":": ".", "NNS": "NOUN", "NNP": "NOUN", "``": ".", "WRB": "ADV", "CC": "CONJ", "LS": "X", "PDT": "DET", "RBS": "ADV", "RBR": "ADV", "CD": "NUM", "EX": "DET", 
             "IN": "ADP", "WP$": "PRON", "MD": "VERB", "NNPS": "NOUN", "JJS": "ADJ", "JJR": "ADJ", "SYM": "X", "VB": "VERB", "UH": "X", "ROOT-POS": "ROOT-CPOS", 
             "-LRB-": ".", "-RRB-": "."}
