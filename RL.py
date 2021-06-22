# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:59:17 2017

@author: faraz
Based on "final_cluster_2.7_RL" code
"""
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from random import randint
import json
import cProfile, pstats, io
import time
matplotlib.use('Agg') # to shut the cluster up about DISPLAY error

#########################################FUNCTIONS################################################

#To ball opinions around a main case, based on distance matrix built using the texts and not the
#citations (just topic modeling)###################################################################   
def Ball(case, Dist_M, ballSize,citation_ct):
    ball=Dist_M[case].filter(items=citation_ct.keys()).nsmallest(ballSize+1)[1:]
    #It returns the opinion_labels of the ranked distance-vector ("ball_tm"), which constructs a ball
    #of opinions around the given case. Should be mentioned, it is just based on the topic modeling
    #information.                    
    return ball
    #Returns a list of the ball_tm's opinion_labels, which represents the ball of opinions around the 
    #given case.
    
######
def Normalize(serie):
    return serie/sum(serie)

    
###########Learning Function-(sigma-greedy SARSA)############
def Update(S,State_Action_M,Dis,Textual_SIM_Dis,OldNode,Main_node,ball,Size,citation_count,Error,sigma_greedy_percentage,dis_rate,weights,P_Reward, N_Reward,greedy_count):
    #To find best action
    if OldNode==Main_node: #If using Textual_SIM_Dis
        D=Textual_SIM_Dis
        print 
    else:
        D=Dis #IF using Dis

        
    Qs_=State_Action_M[S[0],S[1],S[2],S[3],0:].tolist()  #it reads the state-action values based on the given state
    if S[2]==0 and S[3]==0:   #if it is starting a new issue
        BestActions=[ i+1 for i,v in enumerate(Qs_[1:]) if v==max(Qs_[1:]) ]   #then we skip "go back to main node to start a new issue" action 
        if len(BestActions)>1: #If there is more than 1 BestAction
            BestAction=random.choice(BestActions) 
        if random.randint(1, 100)<sigma_greedy_percentage:   ##sigma-greediness part##
            action=random.randint(1, 2)
            greedy_count=greedy_count+1
        else:
            action=BestAction 
    else:
        BestActions=[ i for i,v in enumerate(Qs_) if v==max(Qs_) ]
        if len(BestActions)>1: #If there is more than 1 BestAction
            BestAction=random.choice(BestActions)
        if random.randint(1, 100)<sigma_greedy_percentage:   ##sigma-greediness part##
            action=random.randint(0, 2)
            greedy_count=greedy_count+1
        else:
            action=BestAction 
        
    citation_counts_series = pd.Series(citation_count.values(), index=citation_count.keys())
    Normed_ball_Citations_count=Normalize(citation_counts_series.filter(items=ball.index))
    Normed_ball_D=Normalize(D[OldNode].filter(items=ball.index))
    #To update the state, and find the next node Size_of_State_space=[N_of_Cited, Session,N_of_degree_based_in_session, N_of_similarity_based_in_session]
    w1=weights #Ex. (.2, .8)
    w2=weights[::-1]   
    if action==1:#New impact factor based
        S_new=S+(1,0,1,0)
        D1 = w1[0]*Normed_ball_D - w1[1]*Normed_ball_Citations_count   #Suggestion1        
        newnode= D1.nsmallest(1).index[0]
        if unicode(newnode) in citations[Main_node]:
            Reward = P_Reward
            Recall = +1
        else:
            Reward =N_Reward
            Recall = 0
    elif action==2:#New similarity based
        S_new=S+(1,0,0,1)
        D2 = w2[0]*Normed_ball_D - w2[1]*Normed_ball_Citations_count   #Suggestion1
        newnode= D2.nsmallest(1).index[0]
        if unicode(newnode) in citations[Main_node]:
            Reward = P_Reward
            Recall = +1
        else:
            Reward = N_Reward
            Recall = 0
    elif action==0: #New session
        S_new = S + (0,1,0,0)
        S_new[3] = 0
        S_new[2] = 0
        newnode=Main_node 
        Reward = 0
        Recall = 0
    
    try:
        StateValue=State_Action_M[S[0],S[1],S[2],S[3],action] #existing Q(old_S, selected action)
        Qs_new=State_Action_M[S_new[0],S_new[1],S_new[2],S_new[3],0:].tolist() #best action in next state=Q(new state,best action)
        NewStateValue =StateValue+dis_rate*(Reward+max(Qs_new)-StateValue)
        State_Action_M[S[0],S[1],S[2],S[3],action]=NewStateValue
    except:
        Error=Error+1

    return S_new, newnode, State_Action_M, Error,Reward,Recall,greedy_count


def cite(main_case, S,State_Action_M,Dis,Textual_SIM_Dis,OldNode,ball,Size,citation_count,weights):
    #To find best action
    if OldNode==main_case:
        D=Textual_SIM_Dis
    else:
        D=Dis
    Qs_=State_Action_M[S[0],S[1],S[2],S[3],0:].tolist()  #it reads the state-action values based on the given state
    if S[2]==0 and S[3]==0:   #if it is starting a new issue
        BestActions=[ i+1 for i,v in enumerate(Qs_[1:]) if v==max(Qs_[1:]) ]   #then we skip "go back to main node to start a new issue" action 
        if len(BestActions)>1:
            BestAction=random.choice(BestActions) 
    else:
        BestActions=[ i for i,v in enumerate(Qs_) if v==max(Qs_) ]
        if len(BestActions)>1:
            BestAction=random.choice(BestActions)
    action=BestAction    
    citation_counts_series = pd.Series(citation_count.values(), index=citation_count.keys())
    Normed_ball_Citations_count=Normalize(citation_counts_series.filter(items=ball.index))
    Normed_ball_D=Normalize(D[OldNode].filter(items=ball.index))
    #To update the state, and find the next node Size_of_State_space=[N_of_Cited, Session,N_of_degree_based_in_session, N_of_similarity_based_in_session]
    w1=weights #(.2, .8)
    w2=weights[::-1] #(.8, .2)      
    if action==1:   #New impact factor based
        S_new=S+(1,0,1,0)
        D1 = w1[0]*Normed_ball_D - w1[1]*Normed_ball_Citations_count   #Suggestion1        
        newnode= D1.nsmallest(1).index[0]
        if unicode(newnode) in citations[main_case]:
            Recall = +1
        else:
            Recall = 0
    elif action==2:  #New similarity based
        S_new=S+(1,0,0,1)
        D2 = w2[0]*Normed_ball_D - w2[1]*Normed_ball_Citations_count   #Suggestion1
        newnode= D2.nsmallest(1).index[0]
        if unicode(newnode) in citations[main_case]:
            Recall = +1
        else:
            Recall = 0
    elif action==0: #New session
        S_new = S + (0,1,0,0)
        S_new[3] = 0
        S_new[2] = 0
        newnode=main_case 
        Recall = 0
    
    return S_new, newnode, Recall

def cover(main_case,already_cited,Dis,Textual_SIM_Dis,ball,citation_count,weights):      
    citation_counts_series = pd.Series(citation_count.values(), index=citation_count.keys())
    Normed_ball_Citations_count=Normalize(citation_counts_series.filter(items=ball.index))
    Normed_ball_D=Normalize(Textual_SIM_Dis[main_case].filter(items=ball.index))
    already_cited_Dis=[]
    if len(already_cited)>0:
        for case in ball.index:
            already_cited_Dis.append(sum(Dis[case].filter(items=already_cited)))
        zibra=pd.Series(already_cited_Dis, index=ball.index)
        zib2=Normalize(zibra)
    else:
        zib2=Normed_ball_D
        weights[2]=0
    D1 = weights[0]*Normed_ball_D - weights[1]*Normed_ball_Citations_count-weights[2]*zib2   #Suggestion1        
    newnode= D1.nsmallest(1).index[0]
    if unicode(newnode) in citations[main_case]:
        Recall = +1
    else:
        Recall = 0
    
    return newnode, Recall
    
#used to read back the constructed data in the first part of the code to be used in the second part    
def load_from(name,path):
    with open (path+"/"+name) as f:
        return json.load(f)



pr0 = cProfile.Profile()
pr0.enable()

#To keep track of the running time
Starttime=time.time()





"""setting working dir path and other env veriables"""
dir_path = os.getcwd()
#running on any platform, if you set this address right other paths will be generated automatically
os.chdir(dir_path)



"""setting the paths for source data to be read from"""
#algorithm  is feeded using the SCOTUS bulk data downloaded from Courtlistener.com to Rawdata_path in the following seperate folders:
input_path =  os.path.join("RL input Data","Data")


#To build the citation vectors
R_opiniondatas_number=load_from ("opiniondatas_number",input_path)
R_opiniondatas_names=load_from ("opiniondatas_names",input_path)


in_network_SCOTUS_timespan_opinions_list_zz=load_from ("in_network_SCOTUS_timespan_opinions_list_z",input_path)

RR_cited_the_timespan_SCOTUS_corpus_vectors=load_from ("cited_the_timespan_SCOTUS_corpus_vectors",input_path)

RR_cited_by_timespan_SCOTUS_corpus_vectors=load_from("cited_by_timespan_SCOTUS_corpus_vectors", input_path)

#cited_the_timespan_SCOTUS_corpus_vectors_count_zz={}
cited_by_timespan_SCOTUS_corpus_count_zz={}
for op in RR_cited_by_timespan_SCOTUS_corpus_vectors:
    if len(RR_cited_by_timespan_SCOTUS_corpus_vectors[op])>0:
        cited_by_timespan_SCOTUS_corpus_count_zz[op]=len(RR_cited_by_timespan_SCOTUS_corpus_vectors[op])


opinion_labels=in_network_SCOTUS_timespan_opinions_list_zz
R_DTM_title_list_zz=load_from ("DTM_title_list_z",input_path)

    
 
citations=RR_cited_the_timespan_SCOTUS_corpus_vectors
    
    
n=len(opinion_labels)                                    #size of the corpus (size of the set of the opinions)
#opinion_labels=[str(k+1) for k in range(n)]             #list of opinions

#VERSION2 to build Distance matrix-reading from DIST matrix
#D_in_array_format = genfromtxt('DIST.csv', delimiter=',')
#D_in_array_format = D_in_array_format[:10346,:10346]        #because of lack of the data-Should be gegt corrected
#D_tm=pd.DataFrame(data=genfromtxt('DIST.csv', delimiter=','), index=opinion_labels, columns=opinion_labels)
#D_tm3=pd.read_csv("DIST.csv",names=opinion_labels, index_col=opinion_labels)
D_tm=pd.read_csv('DIST.csv',names=opinion_labels).set_index(pd.Index(opinion_labels))
#D=D_tm.set_index(pd.Index(opinion_labels))

 #doesn't have citation information
#D_original=pd.read_csv('DIST.csv')

#pd.read_csv("DIST.csv",header=opinion_labels, names=["a","b"], index_col=opinion_labels)
Textual_SIM_Dis=pd.read_csv("Textual_SIM.csv",names=opinion_labels).set_index(pd.Index(opinion_labels))

#Textual_SIM_Dis=pd.DataFrame(data=genfromtxt('Textual_SIM.csv', delimiter=','), index=opinion_labels, columns=opinion_labels)


#Distance matrix, considering only topic modelin
##############################################################################################################################################################################################################################################################################                 
main_case=unicode(134735)
#main_case=unicode(127926)

#############################################################################################################################################################################################################################################################################
#Default ball size###########################################################################################################################################################################################################################################################
N_of_Cited=50
ballSize= 750  ###############################################################################################################################################################################################################################################################

#############################################################################################################################################################################################################################################################################
#Learning parameters:########################################################################################################################################################################################################################################################
weights=(0.4,0.6)
#len(citations[main_case])/3  #=25                                
#Number of citations required
sigma_greedy_percentage=35


I=1000
gap=I/20
dis_rate=0.40
P_Reward = +1
N_Reward = -0.5###############################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################################################

ball_tm = Ball(main_case,Textual_SIM_Dis,2*ballSize,cited_by_timespan_SCOTUS_corpus_count_zz) #It returns the opinion_labels of the ranked dist-vector which is the set of opinions 
test_ball=[]
train_ball=[]                                                             #in the opinion ball of the main opinion. 

for case in ball_tm.index:
    if citations.has_key(case) and len(citations[case])>(len(citations[main_case])/3):# and len(citations[case])<(len(citations[main_case])+20):
        if random.randint(0,100)<20:
            test_ball.append(case)
        else:
            train_ball.append(case)
                                                                    
                                                               
                                                               
                                                               
#I=55000                                                     
#Number of iterations to each learn session

#Building state space+Q(s,a)
#N_of_Cited=len(citations[main_case])/3  #=25                                
#Number of citations required
Session=N_of_Cited+1
N_of_degree_based_in_session=N_of_Cited+1
N_of_similarity_based_in_session=N_of_Cited+1
Actions=3
Size_of_State_space=[N_of_Cited+1, Session,N_of_degree_based_in_session, N_of_similarity_based_in_session]     #Size_of_the the state_space
#State_Matrix=np.empty((N_of_Cited, Session, N_of_degree_based_in_session,N_of_similarity_based_in_session))  # 3D array
State_Action_M= np.zeros((N_of_Cited, Session, N_of_degree_based_in_session,N_of_similarity_based_in_session,Actions))  # 4D array
Recallvector=[]
Ranadom_walk_Rewards=[]
citations_count_should_have_been_retrived=[]
A_citations_count_should_have_been_retrived=[]
A_Recallvector=[]
A2_Recallvector=[]
future=now=0
for i in range(I):    
   Trecall=0 
   m=train_ball[randint(0,len(train_ball)-1)]
   Main_node=m
   candidates_ball=Ball(Main_node,Textual_SIM_Dis,ballSize+100,cited_by_timespan_SCOTUS_corpus_count_zz)  #serie of opinions in a ball
   date_adjusted_candidates_for_the_ball=[a for a in candidates_ball.index if R_opiniondatas_number[a+".json"]<R_opiniondatas_number[Main_node+".json"]]
   Temp_ball_tm = candidates_ball.filter(items=date_adjusted_candidates_for_the_ball)[:ballSize]
   S=np.array([0,1,0,0])
   Error=0
   OldNode=Main_node
   count = 0
   #Random-Walk###############################################################################################################################################################################################################################################################################
   ###########################################################################################################################################################################################################################################################################################
   Ranadom_walk_candidates_ball=Ball(Main_node,Textual_SIM_Dis,ballSize+100,cited_by_timespan_SCOTUS_corpus_count_zz)  #serie of opinions in a ball
   Ranadom_walk_date_adjusted_candidates_for_the_ball=[a for a in Ranadom_walk_candidates_ball.index if R_opiniondatas_number[a+".json"]<R_opiniondatas_number[Main_node+".json"]]
   Ranadom_walk_Temp_ball_tm = Ranadom_walk_candidates_ball.filter(items=Ranadom_walk_date_adjusted_candidates_for_the_ball)[:ballSize]
   Ranadom_walk_citations=Ranadom_walk_Temp_ball_tm.nsmallest(N_of_Cited)
   Ranadom_walk_Rewards.append(len(set(Ranadom_walk_citations.index)&set(citations[Main_node])))
   
   #Learning Process##########################################################################################################################################################################################################################################################################
   ###########################################################################################################################################################################################################################################################################################
   greedy_count=0
   while (S[0] < N_of_Cited):   #number of opinions we want to cite
       S_new, newnode, State_Action_M, Error, Reward,Recall, greedy_count = Update(S,State_Action_M,D_tm,Textual_SIM_Dis,OldNode,Main_node,Temp_ball_tm, Size_of_State_space,cited_by_timespan_SCOTUS_corpus_count_zz, Error,sigma_greedy_percentage,dis_rate,weights, P_Reward, N_Reward,greedy_count)
       Trecall=Trecall+Recall
       S = S_new
       OldNode = newnode
       if OldNode!= Main_node:
           del Temp_ball_tm[OldNode]
       
   Recallvector.append(Trecall)
   citations_count_should_have_been_retrived.append((len(citations[Main_node])))
   
   #Actuall citation##########################################################################################################################################################################################################################################################################
   ###########################################################################################################################################################################################################################################################################################
   ###########################################################################################################################################################################################################################################################################################
   if future==now:
       future=now+(gap/2)   
       T_expected_citations=[]
       T_A_Recall=0
       for main_c in test_ball:   
           A_candidates_ball=Ball(main_c,Textual_SIM_Dis,ballSize+100,cited_by_timespan_SCOTUS_corpus_count_zz)  #serie of opinions in a ball
           A_date_adjusted_candidates_for_the_ball=[a for a in A_candidates_ball.index if R_opiniondatas_number[a+".json"]<R_opiniondatas_number[main_c+".json"]]
           A_Temp_ball_tm = A_candidates_ball.filter(items=A_date_adjusted_candidates_for_the_ball)[:ballSize]
           A_S=np.array([0,1,0,0])
           A_OldNode=main_c
           cited_by_RL=[]
           T_expected_citations.append((len(citations[main_c]))) 
           while (A_S[0] < N_of_Cited):  #number of opinions we want to cite
               A_S_new, A_newnode, A_Recall = cite(main_c, A_S,State_Action_M,D_tm,Textual_SIM_Dis,A_OldNode,A_Temp_ball_tm, Size_of_State_space,cited_by_timespan_SCOTUS_corpus_count_zz,weights)
               T_A_Recall=T_A_Recall+A_Recall
               if A_newnode!=main_c:
                   cited_by_RL.append(A_newnode) 
               A_OldNode = A_newnode
               if A_OldNode!= main_c:
                   del A_Temp_ball_tm[A_OldNode]
               A_S = A_S_new
   A_Recallvector.append(float(T_A_Recall)/len(test_ball))
   A_citations_count_should_have_been_retrived.append(float(sum(T_expected_citations))/len(test_ball))
   now=now+1
    
    


#Actuall Random walk for the main case
covering_weights=[.6,.3,.1]
A_Ranadom_walk_Rewards_inball=[]
#Actuall Random walk and Covering Citation for the main case
Covering_Rewards_inball=[]
for main_c in test_ball:
    A_Ranadom_walk_candidates_ball=Ball(main_c,Textual_SIM_Dis,ballSize+100,cited_by_timespan_SCOTUS_corpus_count_zz)  #serie of opinions in a ball
    A_Ranadom_walk_date_adjusted_candidates_for_the_ball=[a for a in A_Ranadom_walk_candidates_ball.index if R_opiniondatas_number[a+".json"]<R_opiniondatas_number[main_c+".json"]]
    A_Ranadom_walk_Temp_ball_tm = A_Ranadom_walk_candidates_ball.filter(items=A_Ranadom_walk_date_adjusted_candidates_for_the_ball)[:ballSize]
    A_Ranadom_walk_citations=A_Ranadom_walk_Temp_ball_tm.nsmallest(N_of_Cited)
    A_Ranadom_walk_Rewards_inball.append(len(set(A_Ranadom_walk_citations.index)&set(citations[main_c])))
    cover_ball=A_Ranadom_walk_candidates_ball.filter(items=A_Ranadom_walk_date_adjusted_candidates_for_the_ball)[:ballSize]
    already_cited=[]
    cited_by_Covering=[]
    C_Recallvector=[]
    T_C_Recall=0
    while (len(already_cited) < N_of_Cited):
        C_newnode, C_Recall = cover(main_c,already_cited,D_tm,Textual_SIM_Dis,cover_ball,cited_by_timespan_SCOTUS_corpus_count_zz,covering_weights)
        T_C_Recall=T_C_Recall+C_Recall
        already_cited.append(C_newnode)
        del cover_ball[C_newnode]
        if C_Recall==1:
            C_Recallvector.append(C_newnode)
    Covering_Rewards_inball.append(T_C_Recall)
Covering_Rewards=(float(sum(Covering_Rewards_inball)))/len(test_ball)
A_Ranadom_walk_Rewards=(float(sum(A_Ranadom_walk_Rewards_inball)))/len(test_ball)



pr0.disable()
s = io.StringIO()
sortby = 'cumulative'
ps0 = pstats.Stats(pr0, stream=s).sort_stats(sortby)
pstats.Stats(pr0).print_stats()
print(s.getvalue())

timestamp=str(time.strftime('%X')[:5])#+" "+time.strftime('%x')[:5])

"""
###########################################fig1-Learning-recall#####################################################################################################################################################################################################################################################
"""
RL_Recall_at_n=[]
Random_walk_Recall_at_n=[]
for i in range(len(Recallvector)):
    RL_Recall_at_n.append(float(Recallvector[i])/citations_count_should_have_been_retrived[i])
    Random_walk_Recall_at_n.append(float(Ranadom_walk_Rewards[i])/citations_count_should_have_been_retrived[i])

fs=12 #fontsize
Ret=[]
RetVar=[]    
n=m=0
#gap=5000
Random_walk_Ret=[]
Retdif=[]
while n<I:
    m=n+gap
    Ret.append(float(sum(RL_Recall_at_n[n:m]))/gap)
    Random_walk_Ret.append(float(sum(Random_walk_Recall_at_n[n:m]))/gap)
    Retdif.append((float(sum(RL_Recall_at_n[n:m]))/gap)-(float(sum(Random_walk_Recall_at_n[n:m]))/gap))
    RetVar.append(np.var(RL_Recall_at_n[n:m]))
    #print n,m
    n=m



fig1 = plt.figure(figsize=(23,9))

x=np.arange(len(Ret))
y=Ret
y2=Random_walk_Ret
z=Retdif

slope1, b = np.polyfit(x, y, 1)
slope2, b2 =np.polyfit(x, y2, 1)
slope_d, b3 = np.polyfit(x, z, 1)

##################################### 
ax1 = plt.subplot(221)
plt.plot(x, y, 'v')
plt.plot(x, slope1*x + b, '-')
plt.plot(x, y2, 'rs')
plt.plot(x, slope2*x + b2, '-')
plt.title("RL Recall@"+str(N_of_Cited)+" and "+"Random walk Recall@"+str(N_of_Cited),fontsize=fs)
plt.ylabel('Recall @ '+str(N_of_Cited))

Recall_diff_slope=slope1


#####################################
ax2 = plt.subplot(223)
plt.plot(x, z, 'v')
plt.plot(x, slope_d*x + b3, '-')
plt.title("RL Recall@"+str(N_of_Cited)+" - "+"Random walk Recall@"+str(N_of_Cited),fontsize=fs-2)
plt.ylabel("Recall @ "+str(N_of_Cited)+" difference")

Recl=[]  
ReclVar=[] 
Random_walk_Recl=[] 
Recdiff=[]
n=m=0
while n<I:
    m=n+gap
    Recl.append(float(sum(Recallvector[n:m]))/gap)
    Random_walk_Recl.append(float(sum(Ranadom_walk_Rewards[n:m]))/gap)
    Recdiff.append((float(sum(Recallvector[n:m]))/gap)-(float(sum(Ranadom_walk_Rewards[n:m]))/gap))
    ReclVar.append(np.var(RL_Recall_at_n[n:m]))
    n=m



##########################################################################
##########################################################################    
x=np.arange(len(Recl))
y=Recl
#y2=ReclVar
y2=Random_walk_Recl
z=Recdiff

slope1, b = np.polyfit(x, y, 1)
slope2, b2 =np.polyfit(x, y2, 1)
slope_d, b_d = np.polyfit(x, z, 1)





#####################################
ax3 = plt.subplot(222)
plt.title("Recalls count for R-Learning process and random walk ",fontsize=fs)
plt.plot(x, y, 'v')
plt.plot(x, slope1*x + b, '-')
plt.plot(x, y2, 'rs')
plt.plot(x, slope2*x + b2, '-')
plt.ylabel('Recall rate')




#####################################
ax4 = plt.subplot(224)
plt.ylabel('Recall  difference')
plt.title("Recall count(Learning process) - Recall count(random walk)",fontsize=fs-2)
plt.plot(x, z, 'v')
plt.plot(x, slope_d*x + b_d, '-')
plt.ylabel('Recall count difference')



fig1.text(
        .92, .42,"\nMain case is: "+str(main_case)+\
        "\nNum of Citations: "+str(N_of_Cited)+\
        "\nTrain ball size: "+str(len(train_ball))+\
        "\nTest ball size: "+str(len(train_ball))+\
        "\nSlop: "+str(slope1)[:7]+str(slope1)[-4:]+\
        "\nSlop diff: "+str(slope_d)[:7]+str(slope_d)[-4:]+\
        "\nEpisodes= "+ str(I)+"\nBall size="+ str(ballSize)+\
        "\nSig greedy rate="+ str(sigma_greedy_percentage)+\
        "\nDiscount rate="+str(dis_rate)+\
        "\nP Reward= "+str(P_Reward)+ "\nN Reward="+ str(N_Reward)+\
        "\nw="+ str(weights)+"\ncoveringw="+ str(covering_weights)+\
        "\npassed time= "+str(time.strftime("%H:%M:%S", time.gmtime(time.time()-Starttime))),        
        ha='left')


fig1.savefig("./134735/"+str(main_case)+" train@"+str(N_of_Cited)+" learning graph after "+str(I)+" eprisodes in a ball of "+str(N_of_Cited)+" cases"+" ID:"+str(random.randint(1,1000))+".png", dpi=500)
plt.clf()


    
"""
###########################################fig2#####################################################################################################################################################################################################################################################
#######################################Actual-Recall#####################################################################################################
""" 

A_RL_Recall_at_n=[]
A_Random_walk_Recall_at_n=[]
A_Covering_Recall_at_n=[]

for i in range(len(A_Recallvector)):
    A_RL_Recall_at_n.append(float(A_Recallvector[i])/A_citations_count_should_have_been_retrived[i])
    A_Random_walk_Recall_at_n.append(float(A_Ranadom_walk_Rewards)/A_citations_count_should_have_been_retrived[i])
    A_Covering_Recall_at_n.append(float(Covering_Rewards)/A_citations_count_should_have_been_retrived[i])



fs=12 #fontsize
A_Ret=[]
A_RetVar=[]    
n=m=0
#gap=5000
Random_walk_A_Ret=[]
covering_A_Ret=[]


A_Retdif=[]
while n<I:
    m=n+gap
    A_Ret.append(float(sum(A_RL_Recall_at_n[n:m]))/gap)
    Random_walk_A_Ret.append(float(sum(A_Random_walk_Recall_at_n[n:m]))/gap)
    covering_A_Ret.append(float(sum(A_Covering_Recall_at_n[n:m]))/gap)
    A_Retdif.append((float(sum(A_RL_Recall_at_n[n:m]))/gap)-(float(sum(A_Random_walk_Recall_at_n[n:m]))/gap))
    A_RetVar.append(np.var(A_RL_Recall_at_n[n:m]))
    #print n,m
    n=m



fig = plt.figure(figsize=(35,14))

x=np.arange(len(A_Ret))
y=A_Ret
y2=Random_walk_A_Ret
y3=covering_A_Ret
z=A_Retdif



A_Ret_recall=A_Ret
Random_walk_A_Ret_recall=Random_walk_A_Ret
covering_A_Ret_recall=covering_A_Ret

slope1, b = np.polyfit(x, y, 1)
slope2, b2 =np.polyfit(x, y2, 1)
slope3, b3 =np.polyfit(x, y3, 1)
slope_d, b_d = np.polyfit(x, z, 1)


##################################### 
ax1 = plt.subplot(231)
plt.plot(x, y, 'v')
plt.plot(x, slope1*x + b, '-')
plt.plot(x, y2, 'rs-')
#plt.plot(x, slope2*x + b2, '-')
plt.plot(x, y3, 'bo-')
#plt.plot(x, slope3*x + b3, '-')

plt.title("Actuall RL Recall and Actual Random walk Recall and Actuall Covering Recall",fontsize=fs)
plt.ylabel('A Recall@'+str(N_of_Cited))


#####################################
ax2 = plt.subplot(234)
plt.plot(x, z, 'v')
plt.plot(x, slope_d*x + b_d, '-')

plt.title("Actuall RL Recall - "+"Actuall Random walk",fontsize=fs-2)
plt.ylabel("A Recall @ "+str(N_of_Cited)+" difference")


##########################################################################
##########################################################################
Recl=[]  
ReclVar=[] 
Random_walk_Recl=[] 
covering_Recl=[]
Recdiff=[]
n=m=0
while n<I:
    m=n+gap
    Recl.append(float(sum(A_Recallvector[n:m]))/gap)
    Random_walk_Recl.append(A_Ranadom_walk_Rewards)
    covering_Recl.append(Covering_Rewards)
    Recdiff.append((float(sum(A_Recallvector[n:m]))/gap)-(A_Ranadom_walk_Rewards))
    ReclVar.append(np.var(A_RL_Recall_at_n[n:m]))
    n=m
    
x=np.arange(len(Recl))
y=Recl
#y2=ReclVar
y2=Random_walk_Recl
z=Recdiff
y3=covering_Recl


slope1, b = np.polyfit(x, y, 1)
slope2, b2 =np.polyfit(x, y2, 1)
slope3, b3 =np.polyfit(x, y3, 1)
slope_d, b_d = np.polyfit(x, z, 1)


#####################################
ax3 = plt.subplot(232)

plt.title("Recall counts for Actuall R-Learning process method and random walk and covering ",fontsize=fs)
plt.plot(x, y, 'v')
plt.plot(x, slope1*x + b, '-')
plt.plot(x, y2, 'rs-')
#plt.plot(x, slope2*x + b2, '-')
plt.plot(x, y3, 'bo-')
#plt.plot(x, slope3*x + b3, '-')
plt.ylabel('A Recall counts@ '+str(N_of_Cited))


#####################################
ax4 = plt.subplot(235)

plt.ylabel('A Recall counts@'+str(N_of_Cited)+' difference')
plt.title("Actuall Recall counts of adaptive method (as Learning process stops) - Actuall Recall countsof the random walk",fontsize=fs-2)
plt.plot(x, z, 'v')
plt.plot(x, slope_d*x + b_d, '-')


"""
#######################################Actual-Precision#####################################################################################################
"""
A_RL_Precision_at_n=[]
A_Random_walk_Precision_at_n=[]
A_Covering_Precision_at_n=[]

for i in range(len(A_Recallvector)):
    A_RL_Precision_at_n.append(float(A_Recallvector[i])/N_of_Cited)
    A_Random_walk_Precision_at_n.append(float(A_Ranadom_walk_Rewards)/N_of_Cited)
    A_Covering_Precision_at_n.append(float(Covering_Rewards)/N_of_Cited)



A_Ret=[]
A_RetVar=[]    
n=m=0
#gap=5000
Random_walk_A_Ret=[]
covering_A_Ret=[]
A_Retdif=[]
while n<I:
    m=n+gap
    A_Ret.append(float(sum(A_RL_Precision_at_n[n:m]))/gap)
    Random_walk_A_Ret.append(float(sum(A_Random_walk_Precision_at_n[n:m]))/gap)
    covering_A_Ret.append(float(sum(A_Covering_Precision_at_n[n:m]))/gap)
    A_Retdif.append((float(sum(A_RL_Precision_at_n[n:m]))/gap)-(float(sum(A_Random_walk_Precision_at_n[n:m]))/gap))
    A_RetVar.append(np.var(A_RL_Precision_at_n[n:m]))
    n=m

x=np.arange(len(A_Ret))
y=A_Ret
y2=Random_walk_A_Ret
y3=covering_A_Ret
z=A_Retdif

slope1, b = np.polyfit(x, y, 1)
slope2, b2 =np.polyfit(x, y2, 1)
slope3, b3 =np.polyfit(x, y3, 1)
slope_d, b_d = np.polyfit(x, z, 1)

##################################### 
ax5 = plt.subplot(233)
plt.plot(x, y, 'v')
plt.plot(x, slope1*x + b, '-')
plt.plot(x, y2, 'rs-')
#plt.plot(x, slope2*x + b2, '-')
plt.plot(x, y3, 'bo-')
#plt.plot(x, slope3*x + b3, '-')

plt.title("Actuall RL Precision and Actual Random walk Precision and Actuall Covering_Precision",fontsize=fs)
plt.ylabel('A Precision@'+str(N_of_Cited))


#####################################
ax6 = plt.subplot(236)
plt.plot(x, z, 'v')
plt.plot(x, slope_d*x + b_d, '-')

plt.title("Actuall RL Precision - Actuall Random walk Precision",fontsize=fs-2)
plt.ylabel("A Precision@"+str(N_of_Cited)+" difference")


fig.text(
        .92, .62,"\nMain case is: "+str(main_case)+\
        "\nNum of Citations: "+str(N_of_Cited)+\
        "\nTrain ball size: "+str(len(train_ball))+\
        "\nTest ball size: "+str(len(test_ball))+\
        "\nSlop: "+str(slope1)[:7]+str(slope1)[-4:]+\
        "\nSlop diff: "+str(slope_d)[:7]+str(slope_d)[-4:]+\
        "\nEpisodes= "+ str(I)+"\nBall size="+ str(ballSize)+\
        "\nSig greedy rate="+ str(sigma_greedy_percentage)+\
        "\nDiscount rate="+str(dis_rate)+\
        "\nP Reward= "+str(P_Reward)+ "\nN Reward="+ str(N_Reward)+\
        "\nw="+ str(weights)+"\ncoveringw="+ str(covering_weights)+\
        "\nA_RL Precision@"+str(N_of_Cited)+"=\n"+ str(A_Ret[-1:])[1:7]+\
        "\n\nA Rand walk Precision@"+str(N_of_Cited)+"=\n"+str(Random_walk_A_Ret[-1:])[1:7]+\
        "\n\nA_Cov Precision@"+str(N_of_Cited)+"=\n"+str(covering_A_Ret[-1:])[1:7]+\
        "\nA_RL Recall@"+str(N_of_Cited)+"=\n"+ str(A_Ret_recall[-1:])[1:7]+\
        "\n\nA Rand walk Recall@"+str(N_of_Cited)+"=\n"+str(Random_walk_A_Ret_recall[-1:])[1:7]+\
        "\n\nA_Cov Recall@"+str(N_of_Cited)+"=\n"+str(covering_A_Ret_recall[-1:])[1:7]+\
        "\npassed time= "+str(time.strftime("%H:%M:%S", time.gmtime(time.time()-Starttime))),        
        ha='left')






fig.savefig("./134735/"+str(main_case)+" test@"+str(N_of_Cited)+" perf graph after "+str(I)+" eprisodes in a ball of "+str(N_of_Cited)+" cases"+" ID:"+str(random.randint(1,1000))+".png", dpi=500)
plt.clf()

