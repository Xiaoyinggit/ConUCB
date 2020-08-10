import json
import numpy as np

class SupArm():
    def __init__(self, suparm_id, fv, related_arms):
        self.id=suparm_id
        self.fv=fv
        self.related_arms=related_arms

class SupArmManager:
    def __init__(self, in_folder, Am):
        self.in_folder=in_folder
        self.am=Am
        self.suparms={}
        self.num_suparm=0

    def loadArmSuparmRelation(self):
        fn=self.in_folder+'/arm_suparm_relation.txt'
        tmp_suparms={}
        with open(fn,'r') as fr:
            for line in fr:
                ele=line.strip().split('\t')
                aid=int(ele[0])
                tmp_sams=set()
                se_ele=ele[1].strip(', ').split(',')
                for se in se_ele:
                    se_a=int(se)
                    tmp_sams.add(se_a)
                wei=1.0/len(tmp_sams)

                for sa in tmp_sams:
                    try:
                        tmp=tmp_suparms[sa]
                    except:
                        tmp=tmp_suparms[sa]={}

                    try:
                        tmp=tmp_suparms[sa][aid]
                        raise AssertionError
                    except:
                        tmp_suparms[sa][aid]=wei

                    self.am.arms[aid].suparms[sa]=wei

        # calculate featurevector of suparm
        for sup_a, alist in tmp_suparms.items():
            fv=np.zeros((self.am.dim,1))
            sum_wei=0
            for aid, wei in alist.items():
                fv+=self.am.arms[aid].fv*wei
                sum_wei+=wei
            fv=fv/sum_wei
            self.suparms[sup_a]=SupArm(sup_a,fv,alist)

        self.num_suparm=len(self.suparms)
