from pycnn import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np


class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = Model()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}    
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        
        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.model.add_lookup_parameters("extrn-lookup", (len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.model["extrn-lookup"].init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        if self.bibiFlag:
            self.builders = [LSTMBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model), 
                             LSTMBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model)]
            self.bbuilders = [LSTMBuilder(1, self.ldims * 2, self.ldims, self.model), 
                              LSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [LSTMBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims, self.model), 
                             LSTMBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model), 
                             SimpleRNNBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.model.add_lookup_parameters("word-lookup", (len(vocab) + 3, self.wdims))
        self.model.add_lookup_parameters("pos-lookup", (len(pos) + 3, self.pdims))
        self.model.add_lookup_parameters("rels-lookup", (len(rels), self.rdims))

        self.model.add_parameters("hidden-layer-foh", (self.hidden_units, self.ldims * 2))
        self.model.add_parameters("hidden-layer-fom", (self.hidden_units, self.ldims * 2))
        self.model.add_parameters("hidden-bias", (self.hidden_units))

        self.model.add_parameters("hidden2-layer", (self.hidden2_units, self.hidden_units))
        self.model.add_parameters("hidden2-bias", (self.hidden2_units))

        self.model.add_parameters("output-layer", (1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.model.add_parameters("rhidden-layer-foh", (self.hidden_units, 2 * self.ldims))
            self.model.add_parameters("rhidden-layer-fom", (self.hidden_units, 2 * self.ldims))
            self.model.add_parameters("rhidden-bias", (self.hidden_units))

            self.model.add_parameters("rhidden2-layer", (self.hidden2_units, self.hidden_units))
            self.model.add_parameters("rhidden2-bias", (self.hidden2_units))

            self.model.add_parameters("routput-layer", (len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.model.add_parameters("routput-bias", (len(self.irels)))


    def  __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov  = self.hidLayerFOM * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.outLayer * self.activation(self.hid2Bias + self.hid2Layer * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias)) # + self.outBias
        else:
            output = self.outLayer * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias) # + self.outBias

        return output


    def __evaluate(self, sentence, train):
        exprs = [ [self.__getExpr(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])

        return scores, exprs


    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.routLayer * self.activation(self.rhid2Bias + self.rhid2Layer * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias)) + self.routBias
        else:
            output = self.routLayer * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias) + self.routBias

        return output.value(), output


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                self.hid2Layer = parameter(self.model["hidden2-layer"])
                self.hid2Bias = parameter(self.model["hidden2-bias"])

                self.hidLayerFOM = parameter(self.model["hidden-layer-fom"])
                self.hidLayerFOH = parameter(self.model["hidden-layer-foh"])
                self.hidBias = parameter(self.model["hidden-bias"])

                self.outLayer = parameter(self.model["output-layer"])

                if self.labelsFlag:
                    self.rhid2Layer = parameter(self.model["rhidden2-layer"])
                    self.rhid2Bias = parameter(self.model["rhidden2-bias"])

                    self.rhidLayerFOM = parameter(self.model["rhidden-layer-fom"])
                    self.rhidLayerFOH = parameter(self.model["rhidden-layer-foh"])
                    self.rhidBias = parameter(self.model["rhidden-bias"])

                    self.routLayer = parameter(self.model["routput-layer"])
                    self.routBias = parameter(self.model["routput-bias"])


                for entry in sentence:
                    wordvec = lookup(self.model["word-lookup"], int(self.vocab.get(entry.norm, 0))) if self.wdims > 0 else None
                    posvec = lookup(self.model["pos-lookup"], int(self.pos[entry.pos])) if self.pdims > 0 else None
                    evec = lookup(self.model["extrn-lookup"], int(self.vocab.get(entry.norm, 0))) if self.external_embedding is not None else None
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(sentence, reversed(sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(sentence, reversed(sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(sentence, True)
                heads = decoder.parse_proj(scores) 

                for entry, head in zip(sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                dump = False

                if self.labelsFlag:
                    for modifier, head in enumerate(heads[1:]):
                        scores, exprs = self.__evaluateLabel(sentence, head, modifier+1)
                        sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    yield sentence


    def Train(self, conll_path):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            eeloss = 0.0
            
            self.hid2Layer = parameter(self.model["hidden2-layer"])
            self.hid2Bias = parameter(self.model["hidden2-bias"])

            self.hidLayerFOM = parameter(self.model["hidden-layer-fom"])
            self.hidLayerFOH = parameter(self.model["hidden-layer-foh"])
            self.hidBias = parameter(self.model["hidden-bias"])

            self.outLayer = parameter(self.model["output-layer"])
            if self.labelsFlag:
                self.rhid2Layer = parameter(self.model["rhidden2-layer"])
                self.rhid2Bias = parameter(self.model["rhidden2-bias"])

                self.rhidLayerFOM = parameter(self.model["rhidden-layer-fom"])
                self.rhidLayerFOH = parameter(self.model["rhidden-layer-foh"])
                self.rhidBias = parameter(self.model["rhidden-bias"])

                self.routLayer = parameter(self.model["routput-layer"])
                self.routBias = parameter(self.model["routput-bias"])

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                for entry in sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = lookup(self.model["word-lookup"], int(self.vocab.get(entry.norm, 0)) if dropFlag else 0) if self.wdims > 0 else None
                    posvec = lookup(self.model["pos-lookup"], int(self.pos[entry.pos])) if self.pdims > 0 else None
                    evec = None
                    
                    if self.external_embedding is not None:
                        evec = lookup(self.model["extrn-lookup"], self.vocab.get(entry.norm, 0) if (dropFlag or (random.random() < 0.5)) else 0)
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(sentence, reversed(sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(sentence, reversed(sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(sentence, True)
                gold = [entry.parent_id for entry in sentence]
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:
                    for modifier, head in enumerate(gold[1:]):
                        rscores, rexprs = self.__evaluateLabel(sentence, head, modifier+1)
                        goldLabelInd = self.rels[sentence[modifier+1].relation]
                        wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                        if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                            lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(sentence)

                if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []

                    renew_cg()
            
                    self.hid2Layer = parameter(self.model["hidden2-layer"])
                    self.hid2Bias = parameter(self.model["hidden2-bias"])

                    self.hidLayerFOM = parameter(self.model["hidden-layer-fom"])
                    self.hidLayerFOH = parameter(self.model["hidden-layer-foh"])
                    self.hidBias = parameter(self.model["hidden-bias"])

                    self.outLayer = parameter(self.model["output-layer"])

                    if self.labelsFlag:
                        self.rhid2Layer = parameter(self.model["rhidden2-layer"])
                        self.rhid2Bias = parameter(self.model["rhidden2-bias"])

                        self.rhidLayerFOM = parameter(self.model["rhidden-layer-fom"])
                        self.rhidLayerFOH = parameter(self.model["rhidden-layer-foh"])
                        self.rhidBias = parameter(self.model["rhidden-bias"])

                        self.routLayer = parameter(self.model["routput-layer"])
                        self.routBias = parameter(self.model["routput-bias"])


        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs)))) 
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
