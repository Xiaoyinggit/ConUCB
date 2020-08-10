##input
# -- alg: name of algorithm
# -- input_files: input folder
# -- out_files: output_folder
import os
import json
import argparse
import Arm
import SupArm
import conf
import os
import numpy as np
from LinUCB import LinUCB
from Con_UCB import Con_UCB
import User
import random
import datetime
from multiprocessing import Pool
thread_num=10


class simulateExp:
    def __init__(self, users, arms, suparms,out_folder, pool_size,batchSize=50,noise=None, suparm_noise=None,test_iter=1000, alias='time', dim=50):
        self.users=users
        self.all_arms=arms
        self.suparms=suparms
        self.out_folder=out_folder
        self.noise=noise
        self.suparm_noise=suparm_noise
        self.batchSize=batchSize
        self.poolArticleSize=pool_size
        self.test_iter=test_iter
        self.alias=alias
        self.dim=dim


    def getReward(self,u, arm):
        return np.dot(u.theta.T, arm.fv)

    def regulateArticlePool(self):
		# Randomly generate articles
        all_index=range(0,len(self.all_arms))
        selected_pool_index=np.random.choice(all_index,self.poolArticleSize, replace=False)
        self.armPool={}
        for si in selected_pool_index:
            self.armPool[si]=self.all_arms[si]
        if len(self.armPool) !=self.poolArticleSize:
            raise AssertionError
    def getSuparmReward(self, u, suparm):
        return np.dot(u.theta.T,suparm.fv)

    def getOptimalReward(self,u, article_pool):
        maxReward=float('-inf')
        best_article=None

        for x,x_o in article_pool.items():
            reward=self.getReward(u,x_o)
            if reward> maxReward:
                best_article=x_o
                maxReward=reward
        if best_article==None:
            raise AssertionError
        return maxReward, best_article

    def getL2Diff(self, x, y):
        return np.linalg.norm(x-y) # L2 norm

    def getX(self,arms):
        X_t=np.zeros((len(arms), self.dim))
        i=0
        for aid, ainfo in arms.items():
            X_t[i,:]=ainfo.fv.T
            i+=1
        return X_t

    def getAddiBudget(self, cur_bt, iter_):
        left_budget=-1
        if iter_==0:
             left_budget=cur_bt(iter_)
        else:
            left_budget=cur_bt(iter_)-cur_bt(iter_-1)
            
        if left_budget>0:
            return int(left_budget)
        else:
            return -1


        


    def simulationPerUser(self, u,test_iter):
        process_id=os.getpid()
        print('[simulationPerUser] uid: %d, process_id: %d'%(u.uid, process_id))
        user_regret={}
        theta_diff={}

        user_regret_file=os.path.join(self.out_folder,'users_regret/%d.txt'%u.uid)
        debug_fw = None
        with open(user_regret_file,'w') as fw:
         for iter_ in range(0, test_iter):

             Addi_budget=self.getAddiBudget(algorithms['Arm-Con'].bt, iter_)
             print('[simulationPerUser] uid: %d, iter: %d, addi_budget: %d'%(u.uid, iter_, Addi_budget))
             self.regulateArticlePool()
             cur_iter_noise=self.noise()
             cur_iter_suparm_noise=self.suparm_noise()
             Optimal_Reward, OptimalArticle=self.getOptimalReward(u,self.armPool)
             try:
                 tmp=user_regret[iter_]
                 tmp=theta_diff[iter_]
                 raise AssertionError
             except:
                 user_regret[iter_]={}
                 theta_diff[iter_]={}



             for algname,alg in algorithms.items():
                 pickedArticle=None
                 if algname in ["ConUCB", "Var-RS", "Var-MRC", "Var-LCR"]:
                     tmp_budget=Addi_budget
                     X_t=None
                     if Addi_budget>0:
                         X_t=self.getX(self.armPool)

                     while tmp_budget>0:
                         pickedsuparm=alg.decide_suparms(self.suparms,u.uid,  np.linalg.norm(u.theta),arms=self.armPool, X_t=X_t, debug_fw=debug_fw)
                         reward=self.getSuparmReward(u,pickedsuparm)+cur_iter_suparm_noise
                         alg.updateSuparmParameters(pickedsuparm,reward,u.uid)
                         tmp_budget-=1
                     if Addi_budget>0:
                         alg.increaseSuparmTimes(u.uid)

                 if algname=='Arm-Con':
                     lh_budget=Addi_budget
                     while lh_budget>0:
                         pickedAddiArm=alg.decide(self.armPool, u.uid, np.linalg.norm(u.theta),debug_fw=debug_fw, best_arm=OptimalArticle.id)
                         reward=self.getReward(u,pickedAddiArm)+cur_iter_noise
                         alg.updateParameters(pickedAddiArm,reward, u.uid)
                         lh_budget-=1

                 pickedArticle=None
                 if algname=='Random':
                     pickedIndex=np.random.choice(list(self.armPool.keys()), 1, replace=False)
                     pickedArticle=self.armPool[pickedIndex[0]]
                 else:
                     pickedArticle=alg.decide(self.armPool, u.uid, np.linalg.norm(u.theta),debug_fw=debug_fw, best_arm=OptimalArticle.id)
                 if pickedArticle==None:
                    raise AssertionError
                 reward=self.getReward(u,pickedArticle)+cur_iter_noise
                 if algname!='Random':
                     alg.updateParameters(pickedArticle,reward, u.uid)
                 # calculate regret
                 regret=Optimal_Reward+cur_iter_noise-reward
                 user_regret[iter_][algname]=regret
                 #calculate theta_dff
                 iter_theta_diff=-1
                 if algname!='Random':
                     iter_theta_diff=self.getL2Diff(u.theta,alg.getTheta(u.uid))
                 theta_diff[iter_][algname]=iter_theta_diff
                 fw.write('iter:%d\talgname:%s\tregret:%f\ttheta_diff:%f\n'%(iter_,algname,regret,iter_theta_diff))

        return user_regret, theta_diff


    def runAlgorithms(self, algorithms):
        if self.alias=='time':
            self.starttime=datetime.datetime.now()
            timeRun=self.starttime.strftime('_%m_%d_%H_%M')
            self.alias=timeRun
        out_regret_file=os.path.join(self.out_folder, "AccRegret"+self.alias+'.csv')
        out_theta_file=os.path.join(self.out_folder, "AccTheta"+self.alias+'.csv')

        AlgRegret={}
        AlgThetaD={}
        BatchCumulateRegret={}
        BatchAvgThetaD={}
        for algname, alg in algorithms.items():
            AlgRegret[algname]=[]
            BatchCumulateRegret[algname]=[]

            AlgThetaD[algname]=[]
            BatchAvgThetaD[algname]=[]
        with open(out_regret_file,'w') as fw:
            fw.write('Time(Iteration)\t')
            fw.write(','.join([str(alg_name) for alg_name in algorithms.keys()]))
            fw.write('\n')
        with open(out_theta_file,'w') as fw:
            fw.write('Time(Iteration)\t')
            fw.write(','.join([str(alg_name) for alg_name in algorithms.keys()]))
            fw.write('\n')


        #simulation
        print('[runAlgorithms] Training iterations: %d'%self.test_iter)
        pool= Pool(processes=thread_num)
        results=[]
        for uid, user in self.users.items():
            result=pool.apply_async(self.simulationPerUser,(user,self.test_iter))
            results.append(result)
        pool.close()
        pool.join()
        all_user_regret=[]
        all_theta_diff=[]
        for result in results:
            tmp_regret,tmp_theta_diff=result.get()
            all_user_regret.append(tmp_regret)
            all_theta_diff.append(tmp_theta_diff)
        for iter_ in range(self.test_iter):
            for ure in all_user_regret:
                for algname, reg in ure[iter_].items():
                    AlgRegret[algname].append(reg)
            for ud in all_theta_diff:
                for algname, thetad in ud[iter_].items():
                    AlgThetaD[algname].append(thetad)

            if iter_%self.batchSize==0:
                for alg_name in algorithms.keys():
                    BatchCumulateRegret[alg_name].append(sum(AlgRegret[alg_name]))
                    BatchAvgThetaD[alg_name].append(sum(AlgThetaD[alg_name]))
                    AlgThetaD[alg_name]=[]
                with open(out_regret_file,'a+') as f:
                    f.write(str(iter_)+'\t')
                    f.write(','.join([str(BatchCumulateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')

                with open(out_theta_file,'a+') as f:
                    f.write(str(iter_)+'\t')
                    f.write(','.join([str(BatchAvgThetaD[alg_name][-1]) for alg_name in algorithms.keys()]))
                    f.write('\n')


        finalRegret={}
        for alg_name in algorithms.keys():
            finalRegret[alg_name]=BatchCumulateRegret[alg_name][:-1]
        return finalRegret




def noise():
    return np.random.normal(scale=conf.armNoiseScale)

def suparm_noise():
    return random.gauss(mu=0,sigma=conf.suparmNoiseScale)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--in_folder',dest='in_folder',help='input the folder containing input files')
    parser.add_argument('--out_folder',dest='out_folder', help='input the folder to output')
    parser.add_argument('--poolSize',dest='poolSize',type=int, help='poolSize of each iteration')
    parser.add_argument('--seedIndex',dest='seedIndex',type=int, help='seedIndex')
   

    args=parser.parse_args()

    #load arms
    np.random.seed(conf.seeds_set[args.seedIndex])
    random.seed(conf.seeds_set[args.seedIndex])
    AM=Arm.ArmManager(args.in_folder)
    AM.loadArms()
    print('[main] Finish loading arms: %d'%AM.n_arms)
    #load Suparms
    SAM=SupArm.SupArmManager(args.in_folder,AM)
    SAM.loadArmSuparmRelation()
    print('[main] Finish loading suparms')
    #load User
    UM=User.UserManager(args.in_folder)
    UM.loadUser()
    print('[main] Finishing loading users: %d'%UM.n_user)

    simExperiment=simulateExp(UM.users,AM.arms, SAM.suparms,args.out_folder,args.poolSize,conf.batch_size,noise,suparm_noise,conf.test_iter, alias="time", dim=AM.dim)
    algorithms={}

    algorithms['Random']=None
    algorithms['LinUCB']=LinUCB(AM.dim, conf.linucb_para)
    algorithms['Arm-Con']=LinUCB(AM.dim, conf.linucb_para,bt=conf.bt)
    algorithms['ConUCB']=Con_UCB(AM.dim, conf.conucb_para,'optimal_greedy', bt=conf.bt)


    simExperiment.runAlgorithms(algorithms)
