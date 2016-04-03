class ContingencyTable(object):

    def __init__(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def get_from_predictions(self, predictions, labels):
        self.tp = sum(predictions[labels==1])
        self.tn = sum(1 - predictions[labels==0])
        self.fp = sum(predictions[labels==0])
        self.fn = sum(1 - predictions[labels==1])

    @property
    def pos(self):
        return self.tp + self.fn

    @property
    def neg(self):
        return self.tn + self.fp

    def __str__(self):
        return (('\tPre.+\tPre.-\n'
                 'Act.+\t{}\t{}\n'
                 'Act.-\t{}\t{}').format(self.tp, self.fn, self.fp, self.tn))
