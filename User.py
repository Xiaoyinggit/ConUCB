import numpy as np
import json

class User():
    def __init__(self, uid, theta, pos_review=None, neg_review=None):
        self.uid=uid
        self.theta=theta
        self.pos_review=pos_review
        self.neg_review=neg_review


class UserManager():
    def __init__(self, in_folder):
        self.in_folder=in_folder
        self.users={}
        self.n_user=0


    def loadUser(self):
        self.users={}
        fn=self.in_folder+'/user_preference.txt'
        with open(fn,'r') as fr:
            for line in fr:
                j_s=json.loads(line)
                uid=j_s['uid']
                theta=j_s['preference_v']
                theta_fv=np.array(theta)
                self.users[uid]=User(uid, np.array(theta))
        self.n_user=len(self.users)

    def loadUserWithReview(self):
        self.loadUser()
        # load users' review items
        review_fn=self.in_folder+'/user_review.txt'
        with open(review_fn,'r') as fr:
            for line in fr:
                j_s=json.loads(line)
                uid=j_s['uid']
                try:
                    tmp=self.users[uid]
                except:
                    continue
                    raise AssertionError
                rlist=j_s['review']
                pos_review=[]
                neg_review=[]
                for sp in rlist:
                    if sp[1]<0:
                        neg_review.append(sp)
                    elif sp[1]>0:
                        pos_review.append(sp)
                    else:
                        raise AssertionError

                self.users[uid].pos_review=pos_review
                self.users[uid].neg_review=neg_review
