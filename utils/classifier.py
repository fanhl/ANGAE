from __future__ import print_function
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import roc_curve

class Classifier(object):

    def __init__(self, embeddings,labels):
        self.embeddings = embeddings
        self.Y = np.argmax(labels, -1)

    def train_test_classify(self, train_index, test_index, val_index, seed=1):
        np.random.seed(seed)

        averages = ["micro", "macro"]
        f1s = {}
        #Y = np.argmax(Y, -1)

        X_train = [self.embeddings[x] for x in train_index]
        Y_train = [self.Y[x] for x in train_index]
        X_test = [self.embeddings[x] for x in test_index]
        Y_test = [self.Y[x] for x in test_index]

        clf = LogisticRegression()
        clf.fit(X_train, Y_train)
        Y_  = clf.predict(X_test)


        for average in averages:
            f1s[average]= f1_score(Y_test, Y_, average=average)
        return f1s


    def cross_validation_classify(self, p_labeled=0.1, n_repeat=10, norm=False):
        """
        Train a classifier using the node embeddings as features and reports the performance.

        Parameters
        ----------
        features : array-like, shape [N, L]
            The features used to train the classifier, i.e. the node embeddings
        z : array-like, shape [N]
            The ground truth labels
        p_labeled : float
            Percentage of nodes to use for training the classifier
        n_repeat : int
            Number of times to repeat the experiment
        norm

        Returns
        -------
        f1_micro: float
            F_1 Score (micro) averaged of n_repeat trials.
        f1_micro : float
            F_1 Score (macro) averaged of n_repeat trials.
        """

        #Y = np.argmax(Y, -1)

        lrcv = LogisticRegressionCV()

        if norm:
            self.embeddings = normalize(self.embeddings)

        trace = []
        for seed in range(n_repeat):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
            split_train, split_test = next(sss.split(self.embeddings, self.Y))

            lrcv.fit(self.embeddings[split_train], self.Y[split_train])
            predicted = lrcv.predict(self.embeddings[split_test])

            f1_micro = f1_score(self.Y[split_test], predicted, average='micro')
            f1_macro = f1_score(self.Y[split_test], predicted, average='macro')
            
            trace.append((f1_micro, f1_macro))

        return np.array(trace).mean(0)

    def auc_ap_scores(self,y_true):  
        y_true = y_true.todense()
        y_score = self.predict()
        auc_score = roc_auc_score(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        return auc_score,ap_score
 
    

    def _roc_curve(self,y_true):
        y_true = y_true.todense()
        y_score = self.predict()
        y_true = np.ravel(y_true)
        y_scores = np.ravel(y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return fpr,tpr,thresholds

    def predict(self):
        temp =np.sqrt(np.sum(np.power(self.embeddings,2),axis=-1))
        temp = np.expand_dims(temp,axis=1)
        emb_normal = self.embeddings/temp
        r_adj = np.matmul(emb_normal,emb_normal.T)
        r_adj = r_adj - sp.eye(r_adj.shape[0])
        return r_adj
