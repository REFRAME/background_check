class ContingencyTable(object):

    def __init__(self, tp, tn, fp, fn):
        self.ct = np.asarray([[tp, tn], [fp, fn]])

    def get_from_predictions(self, predictions, labels):
        self.tp = sum(predictions[labels==1])
        self.tn = sum(1 - predictions[labels==0])
        self.fp = sum(predictions[labels==0])
        self.fn = sum(1 - predictions[labels==1])

    @property
    def tp(self):
        return self.ct[0,0]

    @property
    def tn(self):
        return self.ct[0,1]

    @property
    def fp(self):
        return self.ct[1,0]

    @property
    def fn(self):
        return self.ct[1,1]

    @property
    def pos(self):
        return sum(self.ct[:,0])

    @property
    def neg(self):
        return sum(self.ct[:,1])
