import gzip
import json
import numpy as np

class Arm():
    def __init__(self, aid, fv=None, related_suparms={} ):
        self.id=aid
        self.fv=fv
        self.suparms=related_suparms



class ArmManager():
    def __init__(self, in_folder):
        self.in_folder=in_folder
        self.arms={}
        self.n_arms=0
        self.dim=0

    def loadArms(self):
        fn=self.in_folder+'/arm_info.txt'

        with open(fn,'r') as fr:
            for line in fr:
                j_s=json.loads(line)
                aid=j_s['a_id']
                fv=j_s['fv']
                self.dim=len(fv)
                self.arms[aid]=Arm(aid,np.array(fv))

        self.n_arms=len(self.arms)
    
