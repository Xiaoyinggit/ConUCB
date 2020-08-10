import numpy as np
import math

class ConUCB_UserStruct:
    def __init__(self, uid,dim, para, theta=None, tilde_theta=None,init='zero', gtheta_norm=None):
        self.uid=uid
        self.dim=dim
        self.para=para
        self.time=1
        self.suparms_time=1
        self.X_M_tildeM=None
        self.M_tildeM_M=None

        # initialize the feedback on keyword slides
        self.tilde_M=para['tilde_lambda']* np.identity(n=self.dim)
        self.tilde_Y=np.zeros((self.dim,1))
        self.tilde_Minv=np.linalg.inv(self.tilde_M)

        # intialize the feedback on arm slides

        self.M=(1-para['lambda'])*np.identity(n=self.dim)
        self.Y=np.zeros((self.dim,1))
        self.Minv=np.linalg.inv(self.M)

        if init=='random':
            self.tilde_theta=np.random.rand((self.dim,1))
            self.theta=np.random.rand((self.dim,1))
        else:
            self.tilde_theta=np.zeros((self.dim,1))
            self.theta=np.zeros((self.dim,1))

        self.cal_alpha=False
        self.gtheta_norm=gtheta_norm
        self.alpha=-1
        try:
            self.alpha=self.para['alpha']
        except:
            self.cal_alpha=True

        self.tilde_alpha=-1
        try:
            self.tilde_alpha=self.para['tilde_alpha']
        except:
            self.cal_alpha=True
            
        if self.cal_alpha:
            #t1= self.para['lambda']*self.time*25/(self.dim*(1-self.para['lambda']))
            #alpha_t=np.sqrt(self.dim * np.log((1+t1)/self.para['sigma']))
            det_m=np.linalg.det(self.M)
            self.alpha=np.sqrt(2* np.log(math.pow(det_m,1/2)/(math.pow(1-self.para['lambda'], self.dim/2)*self.para['sigma'])))
       
            self.tilde_alpha=(np.sqrt(2*(self.dim *np.log(6)+np.log(2*self.suparms_time/self.para['sigma'])))+2*np.sqrt(self.para['tilde_lambda'])*self.gtheta_norm)



    def get_X_M_tildeM(self,X):
        self.X_M_tildeM=np.dot(np.dot(X,self.Minv),self.tilde_Minv)

    def get_M_tildeM_M(self):
        self.M_tildeM_M=np.dot(np.dot(self.Minv,self.tilde_Minv),self.Minv)
        return self.M_tildeM_M

    def getCredit(self, fv):
        result_a=np.dot(self.X_M_tildeM,fv)
        result_b=1+np.dot(np.dot(fv.T,self.tilde_Minv),fv)
        norm_M=np.linalg.norm(result_a)
        return norm_M*norm_M/result_b

    def getProb(self, fv):
        
        if self.cal_alpha:
            self.tilde_alpha=(np.sqrt(2*(self.dim *np.log(6)+np.log(2*self.suparms_time/self.para['sigma'])))+2*np.sqrt(self.para['tilde_lambda'])*self.gtheta_norm)
        

        if self.alpha==-1 or self.tilde_alpha==-1:
            raise AssertionError

        mean=np.dot(self.theta.T,fv)
        X_M_tM=np.dot(fv.T,self.M_tildeM_M)
        X_M_tM_M_X=np.dot(X_M_tM,fv)
        var1=np.sqrt(np.dot(np.dot(fv.T,self.Minv),fv))
        var2=np.sqrt(X_M_tM_M_X)
        pta=mean+self.para['lambda']*self.alpha*var1+  (1-self.para['lambda'])*self.tilde_alpha*var2
        return pta



    def updateSuparmParameters(self,a_fv, reward):
        self.tilde_Minv=self.getInv(self.tilde_Minv,a_fv)
        self.tilde_M+=np.outer(a_fv,a_fv)
        self.tilde_Y+=a_fv*reward
            
        self.tilde_theta=np.dot(self.tilde_Minv,self.tilde_Y)

        self.theta=np.dot(self.Minv,self.Y+(1-self.para['lambda'])*self.tilde_theta)
    
    def getAvgUncertainty(self,suparm, armPool):
        sum_s=0
        sum_wei=0
        for aid, wei in suparm.related_arms.items():
            a_fv=None
            try:
                a_fv=armPool[aid].fv
            except KeyError:
                continue
            s_a=np.dot(np.dot(a_fv.T, self.M_tildeM_M), a_fv)
            if wei<=0:
                raise AssertionError
            sum_s+=wei*s_a
            sum_wei+=wei

        if sum_wei<0.0001:
            if sum_s>0:
                raise AssertionError
            return sum_s
       
        return sum_s/sum_wei
       
    def getInv(self, old_Minv, nfv):
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

    def updateParameters(self, a_fv, reward):
        self.Minv=self.getInv(self.Minv, np.sqrt(self.para['lambda'])*a_fv)
        self.M+=self.para['lambda']*np.outer(a_fv,a_fv)
        self.Y+=self.para['lambda']*a_fv*reward
        self.theta=np.dot(self.Minv,self.Y+(1-self.para['lambda'])*self.tilde_theta)
        self.time+=1
        
        if self.cal_alpha:
            #t1= self.para['lambda']*self.time*25/(self.dim*(1-self.para['lambda']))
            #alpha_t=np.sqrt(self.dim * np.log((1+t1)/self.para['sigma']))
            det_m=np.linalg.det(self.M)
            self.alpha=np.sqrt(2* np.log(math.pow(det_m,1/2)/(math.pow(1-self.para['lambda'], self.dim/2)*self.para['sigma'])))
       










class Con_UCB:

    def __init__(self, dim, para,suparm_strategy='random',init='zero', bt=lambda t:t+1):
        self.dim=dim
        self.para=para
        self.init=init
        self.suparm_strategy=suparm_strategy
        self.users={}
        self.bt=bt

    def get_suparm_budget(self,uid,norm):
        try:
            tmp=self.users[uid]
        except:
            self.users[uid]=ConUCB_UserStruct(uid,self.dim, self.para, gtheta_norm=norm)

        left_budget=self.bt(self.users[uid].time)- self.bt(self.users[uid].time-1)
        if left_budget>0:
            return int(left_budget)
        else:
            return -1

    def decide_suparms(self, pool_suparm, uid, norm, arms,X_t,debug_fw=None):
        try:
            tmp=self.users[uid]
        except:
            self.users[uid]=ConUCB_UserStruct(uid,self.dim, self.para, gtheta_norm=norm)

        if self.suparm_strategy=='random':
            selected_index=np.random.randint(0,len(pool_suparm)-1)
            #print('[ConUCB_decide_suparms] selected suparm : %d'%selected_index)
            return pool_suparm[selected_index]
        elif self.suparm_strategy=='optimal_greedy':
            picked_suparm=None
            max_C=float('-inf')
            self.users[uid].get_X_M_tildeM(X_t)

            for x, xinfo in pool_suparm.items():
                x_pta=self.users[uid].getCredit(xinfo.fv)
                if x_pta>max_C:
                    picked_suparm=xinfo
                    max_C=x_pta
            return picked_suparm
        elif self.suparm_strategy=='uncertain':
            picked_suparm=None
            max_C=float('-inf')
            self.users[uid].get_M_tildeM_M()
            for x, xinfo in pool_suparm.items():
                tilde_s_k=self.users[uid].getAvgUncertainty(xinfo,arms)
                if tilde_s_k > max_C:
                    picked_suparm=xinfo
                    max_C=tilde_s_k
            if picked_suparm==None:
                raise AssertionError
            return picked_suparm
        elif self.suparm_strategy=='reduce_more':
            picked_suparm=None
            max_C=float('-inf')
            cur_Minv=self.users[uid].Minv
            cur_tildeMinv=self.users[uid].tilde_Minv
            old_M_tildeM_M=self.users[uid].get_M_tildeM_M()
            for x, xinfo in pool_suparm.items():
                new_tildeMinv=self.users[uid].getInv(cur_tildeMinv,xinfo.fv)
                tilde_s_k=self.getAvgReduction(new_tildeMinv, cur_Minv, arms,xinfo, old_M_tildeM_M, debug_fw)
                if tilde_s_k > max_C:
                    picked_suparm=xinfo
                    max_C=tilde_s_k
            if picked_suparm==None:
                raise AssertionError
            return picked_suparm

        raise AssertionError

                

    def getAvgReduction(self, new_tildeMinv, cur_Minv, arms, xinfo, old_M_tildeM_M, debug_fw=None):
        M_tildeM_M=np.dot(np.dot(cur_Minv,new_tildeMinv), cur_Minv)
        reduce=0
        sum_wei=0
        for aid, wei in xinfo.related_arms.items():
            a_fv=None
            try:
               a_fv=arms[aid].fv
            except:
               continue
            
            new_s_a=np.sqrt(np.dot(np.dot(a_fv.T, M_tildeM_M), a_fv))
            old_s_a=np.sqrt(np.dot(np.dot(a_fv.T, old_M_tildeM_M), a_fv))
            if new_s_a -old_s_a> 0.0001:
                raise AssertionError
            reduce+=wei*(old_s_a-new_s_a)
            sum_wei+=wei
        if sum_wei<0.00001:
            if reduce>0:
                raise AssertionError
            return reduce


        return reduce/sum_wei
            


    def decide(self, pool_arms,uid,norm, debug_fw=None, best_arm=None):
        try:
            tmp=self.users[uid]
        except:
            self.users[uid]=ConUCB_UserStruct(uid,self.dim, self.para, gtheta_norm=norm)

        picked_arm=None
        max_P=float('-inf')


        self.users[uid].get_M_tildeM_M()
        for x, x_o in pool_arms.items():
            x_pta=self.users[uid].getProb(x_o.fv)
            if x_pta>max_P:
                picked_arm=x_o
                max_P=x_pta

        if picked_arm==None:
            raise AssertionError
	

        return picked_arm


    def updateSuparmParameters(self,picked_arm,reward, uid):
        self.users[uid].updateSuparmParameters(picked_arm.fv, reward)

    def updateParameters(self, picked_arm,reward,uid):
        self.users[uid].updateParameters(picked_arm.fv, reward)

    def getTheta(self, uid):
        return self.users[uid].theta

    def increaseSuparmTimes(self, uid):
        self.users[uid].suparms_time+=1

