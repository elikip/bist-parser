from dynet import *
from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np


class ArcHybridLSTM:
    def __init__(self, words, pos, rels, w2i, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        random.seed(1)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.oracle = options.oracle
        self.ldims = options.lstm_dims * 2
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.window

        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)

        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        dims = self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)
        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag

        if self.bibiFlag:
            self.surfaceBuilders = [VanillaLSTMBuilder(1, dims, self.ldims * 0.5, self.model),
                                    VanillaLSTMBuilder(1, dims, self.ldims * 0.5, self.model)]
            self.bsurfaceBuilders = [VanillaLSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model),
                                     VanillaLSTMBuilder(1, self.ldims, self.ldims * 0.5, self.model)]
        elif self.blstmFlag:
            if self.layers > 0:
                self.surfaceBuilders = [VanillaLSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model), LSTMBuilder(self.layers, dims, self.ldims * 0.5, self.model)]
            else:
                self.surfaceBuilders = [SimpleRNNBuilder(1, dims, self.ldims * 0.5, self.model), LSTMBuilder(1, dims, self.ldims * 0.5, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.word2lstm = self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.word2lstmbias = self.model.add_parameters((self.ldims))
        self.lstm2lstm = self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.lstm2lstmbias = self.model.add_parameters((self.ldims))

        self.hidLayer = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((3, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias = self.model.add_parameters((3))

        self.rhidLayer = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.rhidBias = self.model.add_parameters((self.hidden_units))

        self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

        self.routLayer = self.model.add_parameters((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias = self.model.add_parameters((2 * (len(self.irels) + 0) + 1))


    def __evaluate(self, stack, buf, train):
        topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [self.empty] for i in xrange(self.k) ]
        topBuffer = [ buf.roots[i].lstms if len(buf) > i else [self.empty] for i in xrange(1) ]

        input = concatenate(list(chain(*(topStack + topBuffer))))

        if self.hidden2_units > 0:
            routput = (self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(self.rhidLayer.expr() * input + self.rhidBias.expr())) + self.routBias.expr())
        else:
            routput = (self.routLayer.expr() * self.activation(self.rhidLayer.expr() * input + self.rhidBias.expr()) + self.routBias.expr())

        if self.hidden2_units > 0:
            output = (self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(self.hidLayer.expr() * input + self.hidBias.expr())) + self.outBias.expr())
        else:
            output = (self.outLayer.expr() * self.activation(self.hidLayer.expr() * input + self.hidBias.expr()) + self.outBias.expr())

        scrs, uscrs = routput.value(), output.value()

        #transition conditions
        left_arc_conditions = len(stack) > 0 and len(buf) > 0
        right_arc_conditions = len(stack) > 1 and stack.roots[-1].id != 0
        shift_conditions = len(buf) >0 and buf.roots[0].id != 0

        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        if train:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            ret = [ [ (rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2 ] + output1) for j, rel in enumerate(self.irels) ] if left_arc_conditions else [],
                    [ (rel, 1, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels) ] if right_arc_conditions else [],
                    [ (None, 2, scrs[0] + uscrs0, routput[0] + output0) ] if shift_conditions else [] ]
        else:
            s1,r1 = max(zip(scrs[1::2],self.irels))
            s2,r2 = max(zip(scrs[2::2],self.irels))
            s1 += uscrs1
            s2 += uscrs2
            ret = [ [ (r1, 0, s1) ] if left_arc_conditions else [],
                    [ (r2, 1, s2) ] if right_arc_conditions else [],
                    [ (None, 2, scrs[0] + uscrs0) ] if shift_conditions else [] ]
        return ret
        #return [ [ (rel, 0, scrs[1 + j * 2 + 0] + uscrs[1], routput[1 + j * 2 + 0] + output[1]) for j, rel in enumerate(self.irels) ] if len(stack) > 0 and len(buf) > 0 else [],
        #         [ (rel, 1, scrs[1 + j * 2 + 1] + uscrs[2], routput[1 + j * 2 + 1] + output[2]) for j, rel in enumerate(self.irels) ] if len(stack) > 1 else [],
        #         [ (None, 2, scrs[0] + uscrs[0], routput[0] + output[0]) ] if len(buf) > 0 else [] ]


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)

    def Init(self):
        evec = self.elookup[1] if self.external_embedding is not None else None
        paddingWordVec = self.wlookup[1]
        paddingPosVec = self.plookup[1] if self.pdims > 0 else None

        paddingVec = tanh(self.word2lstm.expr() * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2lstmbias.expr() )
        self.empty = paddingVec if self.nnvecs == 1 else concatenate([paddingVec for _ in xrange(self.nnvecs)])


    def getWordEmbeddings(self, sentence, train):
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag =  not train or (random.random() < (c/(0.25+c)))
            root.wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if dropFlag else 0]
            root.posvec = self.plookup[int(self.pos[root.pos])] if self.pdims > 0 else None

            if self.external_embedding is not None:
                #if not dropFlag and random.random() < 0.5:
                #    root.evec = self.elookup[0]
                if root.form in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.form]]
                elif root.norm in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.norm]]
                else:
                    root.evec = self.elookup[0]
            else:
                root.evec = None
            root.ivec = concatenate(filter(None, [root.wordvec, root.posvec, root.evec]))

        if self.blstmFlag:
            forward  = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(sentence, reversed(sentence)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in sentence:
                root.vec = concatenate( [root.fvec, root.bvec] )

            if self.bibiFlag:
                bforward  = self.bsurfaceBuilders[0].initial_state()
                bbackward = self.bsurfaceBuilders[1].initial_state()

                for froot, rroot in zip(sentence, reversed(sentence)):
                    bforward = bforward.add_input( froot.vec )
                    bbackward = bbackward.add_input( rroot.vec )
                    froot.bfvec = bforward.output()
                    rroot.bbvec = bbackward.output()
                for root in sentence:
                    root.vec = concatenate( [root.bfvec, root.bbvec] )

        else:
            for root in sentence:
                root.ivec = (self.word2lstm.expr() * root.ivec) + self.word2lstmbias.expr()
                root.vec = tanh( root.ivec )


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
                self.Init()

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
                self.getWordEmbeddings(conll_sentence, False)
                stack = ParseForest([])
                buf = ParseForest(conll_sentence)

                for root in conll_sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0

                while not (len(buf) == 1 and len(stack) == 0):
                    scores = self.__evaluate(stack, buf, False)
                    best = max(chain(*scores), key = itemgetter(2) )

                    if best[1] == 2:
                        stack.roots.append(buf.roots[0])
                        del buf.roots[0]

                    elif best[1] == 0:
                        child = stack.roots.pop()
                        parent = buf.roots[0]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 0
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                    elif best[1] == 1:
                        child = stack.roots.pop()
                        parent = stack.roots[-1]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 1
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                renew_cg()
                yield sentence


    def Train(self, conll_path):
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ltotal = 0
        ninf = -float('inf')

        hoffset = 1 if self.headFlag else 0

        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)

            errs = []
            eeloss = 0.0

            self.Init()

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
                self.getWordEmbeddings(conll_sentence, True)
                stack = ParseForest([])
                buf = ParseForest(conll_sentence)

                for root in conll_sentence:
                    root.lstms = [root.vec for _ in xrange(self.nnvecs)]

                hoffset = 1 if self.headFlag else 0

                while not (len(buf) == 1 and len(stack) == 0):
                    scores = self.__evaluate(stack, buf, True)
                    scores.append([(None, 3, ninf ,None)])

                    alpha = stack.roots[:-2] if len(stack) > 2 else []
                    s1 = [stack.roots[-2]] if len(stack) > 1 else []
                    s0 = [stack.roots[-1]] if len(stack) > 0 else []
                    b = [buf.roots[0]] if len(buf) > 0 else []
                    beta = buf.roots[1:] if len(buf) > 1 else []

                    left_cost  = ( len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                                   len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[0]) > 0 else 1
                    right_cost = ( len([h for h in b + beta if h.id == s0[0].parent_id]) +
                                   len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[1]) > 0 else 1
                    shift_cost = ( len([h for h in s1 + alpha if h.id == b[0].parent_id]) +
                                   len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id]) )  if len(scores[2]) > 0 else 1
                    costs = (left_cost, right_cost, shift_cost, 1)

                    bestValid = max(( s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == 2 or  s[0] == stack.roots[-1].relation ) ), key=itemgetter(2))
                    bestWrong = max(( s for s in chain(*scores) if costs[s[1]] != 0 or  ( s[1] != 2 and s[0] != stack.roots[-1].relation ) ), key=itemgetter(2))
                    best = bestValid if ( (not self.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong

                    if best[1] == 2:
                        stack.roots.append(buf.roots[0])
                        del buf.roots[0]

                    elif best[1] == 0:
                        child = stack.roots.pop()
                        parent = buf.roots[0]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 0
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                    elif best[1] == 1:
                        child = stack.roots.pop()
                        parent = stack.roots[-1]

                        child.pred_parent_id = parent.id
                        child.pred_relation = best[0]

                        bestOp = 1
                        if self.rlMostFlag:
                            parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                        if self.rlFlag:
                            parent.lstms[bestOp + hoffset] = child.vec

                    if bestValid[2] < bestWrong[2] + 1.0:
                        loss = bestWrong[3] - bestValid[3]
                        mloss += 1.0 + bestWrong[2] - bestValid[2]
                        eloss += 1.0 + bestWrong[2] - bestValid[2]
                        errs.append(loss)

                    if best[1] != 2 and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                        lerrors += 1
                        if child.pred_parent_id != child.parent_id:
                            errors += 1
                            eerrors += 1

                    etotal += 1

                if len(errs) > 50: # or True:
                    #eerrs = ((esum(errs)) * (1.0/(float(len(errs)))))
                    eerrs = esum(errs)
                    scalar_loss = eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    lerrs = []

                    renew_cg()
                    self.Init()

        if len(errs) > 0:
            eerrs = (esum(errs)) # * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
