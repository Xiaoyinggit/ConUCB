import math
linucb_para={'lambda':1,'sigma':0.05, 'alpha': 0.5} 
conucb_para={'lambda':0.5, 'sigma':0.05, 'tilde_lambda':1, 'alpha': 0.25, 'tilde_alpha': 0.25}
train_iter=0
test_iter=1000
armNoiseScale=0.1
suparmNoiseScale=0.1
batch_size=50
bt= lambda t: 5*int(math.log(t+1))
seeds_set=[2756048, 675510, 807110,2165051, 9492253, 927,218,495,515,452]
