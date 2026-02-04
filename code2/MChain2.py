#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import math
import graphviz as gv

class MChain(object): 
    colors = ['#10C937', '#FFFE36', '#5C71F3', '#B6F263', '#C65A37',
              '#9EBBDB', '#9BD055', '#AF8ECD', '#B5B0F9', '#FBACAC',
              '#73DA66', '#F4CD25', '#C4E055', '#5E99C4', '#4EE5C9',
              '#32E6F4', '#B881AD', '#B09C0E', '#B5F6C4', '#FAEA4D',
              '#F3CE3D', '#FEC7FC', '#7CB5D5', '#A9EDAB', '#F80183']

    def __init__(self, device='cpu'):     
        super().__init__()
        self.device = device

    def from_list(self, L, prefix='S'):
        '''
        build chain from a list of transition tuples: (from_state, to_state, probability)

        parameters:
        L: list of transition tuples, probabilities may be unnormalized
        prefix: state names are prefixed (ex: (1,3,0.4) -> produces s1, s3 as state names

        name of state: prefix+state             
        '''
        states = list(set([i0 for i0,_,_ in L] + [j0 for _,j0,_ in L])) 
        N = len(states)
        self.N = N
        self.s = [prefix+str(i0) for i0 in states]
        s2i = {states[i0]:i0 for i0 in range(N)}
        U = torch.zeros(N,N)
        for i0,j0,p0 in L:
            U[s2i[i0],s2i[j0]]=p0
        self.U = U 
        self.S = torch.arange(N, dtype=torch.int)        

        self._initialise()

    def from_matrix(self, U, names=None, prefix='s'):
        '''
        build a chain from a (unnormalized) transition probability matrix

        parameters:        
        U: transition probability matrix
        names: optional list of state names
        prefix: names are build by prefixing the matrix index 
        '''
        self.U = U
        N = U.shape[0]
        self.N = N # number of states
        S = torch.arange(N, dtype=torch.int)
        self.S = S # state indices
        if names is not None:
            s = names           
        else:
            s = [prefix + str(i0) for i0 in range(N)]

        self.s = s # state names

        self._initialise()


    def analyse(self):
        '''
        analyse the chain to get probabilities and expected times
        '''
        torch.set_grad_enabled(False)

        self._process_0()
        self._process_1()

    def subchain(self, states=[], classes=[], absorb=True):
        '''
        build a subchain of states and classes

        parameters:
        states: list of state names
        classes: list of class ids

        returns new chain
        '''
        s = self.s # state names
        P = self.P # tranistion probability matrix
        c = self.c # state classes
        N = self.N       
        s2i = self.s2i

        l0 = [s2i[i0] for i0 in states] + [i0 for i0 in range(N) if c[i0] in classes]
        x0 = torch.tensor([i0 in l0 for i0 in range(N)])
        s0 = [s[i0] for i0 in range(N) if x0[i0]]
        P0 = P[x0][:,x0]

        if absorb:
            u0 = 1 - P0.sum(axis=1)+ torch.diagonal(P0)
            P0 = torch.diagonal_scatter(P0,u0)

        mc1 = MChain()
        mc1.from_matrix(P0, names=s0)

        return mc1

    def union(self, mc1):
        '''
        unify the chain with another chain

        returns new chain        
        '''
        s = self.s # state names

        s0 = s + [i0 for i0 in mc1.s if i0 not in s]        

        N0 = len(s0)

        s2i = {s0[i]:i for i in range(N0)} # map state name to internal index 

        P0 = torch.zeros((N0,N0))

        for i0,j0,p0 in self.list_chain():
            P0[s2i[i0],s2i[j0]] = P0[s2i[i0],s2i[j0]] + p0 
        for i0,j0,p0 in mc1.list_chain():
            P0[s2i[i0],s2i[j0]] = P0[s2i[i0],s2i[j0]] + p0  

        mc2 = MChain()
        mc2.from_matrix(P0, names=s0) 
        return mc2      

    def reduce(self, classes=[], states=[], name=None):
        '''
        states and classes are reduced to one state
        '''
        c = self.c # state classes
        s = self.s # state names
        P = self.P # transition probability matrix      
        N = self.N
        s2i = self.s2i


        if name in s:
            return self 

        l0 = [s2i[i0] for i0 in states] + [i0 for i0 in range(N) if c[i0] in classes]

        x0 = torch.tensor([i0 not in l0 for i0 in range(N)])

        p1 = torch.cat((torch.sum(P[~x0][:,x0],axis=0,keepdims=True),torch.zeros(1,1)),1)

        p2 = P[~x0][:,~x0].sum()

        s0 = [s[i0] for i0 in range(N) if x0[i0]]   
        P0 = P[x0][:,x0]
        n0 = P0.shape[0]

        P0 = torch.cat((P0,1-torch.sum(P0,axis=1,keepdims=True)),1)
        P0 = torch.cat((P0,p1),0)
        P0[n0,n0] = p2

        s2 = s0 + [name]

        mc0 = MChain()
        mc0.from_matrix(P0,s2) 
        return mc0


    def delete(self, classes=[], states=[], absorb=True):
        '''
        delete states or classes from a chain

        absorb==True: the freed probability mass is assigned to the originated state
        absorb==False: the freed probability mass is distributed among the other edges
        '''
        c = self.c # state classes
        s = self.s # state names
        P = self.P # transition probability matrix      
        N = self.N       
        s2i = self.s2i

        l0 = [s2i[i0] for i0 in states] + [i0 for i0 in range(N) if c[i0] in classes]
        x0 = torch.tensor([i0 not in l0 for i0 in range(N)])
        s0 = [s[i0] for i0 in range(N) if x0[i0]]

        P0 = P[x0][:,x0]        

        if absorb:
            u0 = 1 - P0.sum(axis=1)+ torch.diagonal(P0)
            P0 = torch.diagonal_scatter(P0,u0)

        mc0 = MChain()
        mc0.from_matrix(P0, s0)        
        return mc0 

    def display(self, states=[], classes=[], neighbors=True, size=None):
        '''
        display a Markov chain or parts of it

        states:    states included in display
        classe:    states of classes included in display
        neighbors: neighbors of the selected states are included

        if states and classes are empty, the complete chain is displayed
        '''
        c = self.c # state classes
        S = self.S # state ids
        P = self.P # transition probability matrix
        s = self.s # state names
        s2i = self.s2i # maps state names to state ids

        if states or classes:                    
            l0 = states + [s[i0] for i0 in S if c[i0] in classes]
            I0 = torch.tensor([s2i[i0] for i0 in l0], dtype=torch.int)
            if neighbors:
                I1 = S[(P[:,I0]>0).any(axis=1)]
                I2 = S[(P[I0]>0).any(axis=0)]
                I0 = torch.hstack([I0,I1,I2])

            I0 = torch.unique(I0)

            P0 = P[I0][:,I0]

            s0 = [s[i0] for i0 in I0] #s0 = s[I0]
            c0 = c[I0]

            self._show1(s0, P0, c0, size)
        else:
            self._show1(s, P, c, size)           



    def run(self, start=None, steps=100, stop_states=[], stop_classes=[], visits=None, early_stop=True):
        '''
        run the Markov chain 

        returns list of visited states 

        start:        starting state
        steps:        maximal number of steps
        stop_states:  run stops if one of the states is reached
        stop_classes: run stops if one of the classes is reached
        visits:       run stops if stop_statees/classes are visited n times 
        early_stop:   run is terminated when stop_states/classes can not be reached anymore
        '''
        P = self.P # transition probabilty matrix
        S = self.S # state ids
        R = self.R # reachability matrix
        c = self.c # state classes
        s = self.s # state names
        N = self.N


        s2i = self.s2i # maps state name to state id
        i2s = self.i2s # maps state id to state name
        i2j = self.i2j # maps state id to successor ids
        i2p = self.i2p # maps state id to successor probabilities


        l0 = list()

        z1 = [s2i[i0] for i0 in stop_states] + [i0 for i0 in range(N) if c[i0] in stop_classes]

        x0 = R[:,z1].any(axis=1)
        e1 = [i0 for i0 in range(N) if x0[i0]]

        i0 = s2i[start]

        if visits is not None:
            v0 = visits    

        for _ in range(steps):
            if z1 and early_stop and i0 not in e1:                
                break            

            i0 = i2j[i0][torch.multinomial(i2p[i0],1)]

            l0.append(i0)

            if i0 in z1:
                if visits is not None:
                    v0 = v0 - 1
                    if v0<=0:
                        break
                else:
                    break

        return [i2s[i0] for i0 in l0]


    def Hij(self, i0, j0):
        '''
        probability to reach state j0 when starting in state i0
        '''
        return self.H[self.s2i[i0], self.s2i[j0]]


    def Pij(self, i0, j0):
        '''
        probability to jump from state i0 to state j0
        '''
        return self.P[self.s2i[i0], self.s2i[j0]]


    def Rij(self, i0, j0):
        '''
        returns whether there is a path from state i0 to state j0
        '''
        return self.R[self.s2i[i0], self.s2i[j0]] 

    def Eij(self, i0, j0):
        '''
        indicates whether state j0 may be visited from state i0

        0: impossible to visit state j0
        1: must visit state j0 
        -1: may visit state j0
        '''
        return self.E[self.s2i[i0], self.s2i[j0]]    

    def Vij(self, i0, j0):
        '''
        expected visit count of state j0 starting in state i0
        '''
        return self.V[self.s2i[i0], self.s2i[j0]]    

    def Tij(self, i0, j0):
        '''
        time to first visit of state j0 starting in i0
        '''
        return self.T[self.s2i[i0], self.s2i[j0]] 

    def Qi(self, i0):        
        '''
         distribution of absorbing classes starting in i0
        '''
        return self.Q[self.s2i[i0]] 


    def pi(self, i0):
        '''
        period of state i0
        '''
        return self.p[self.s2i[i0]]


    def ci(self, i0):
        '''
        class of state i0
        '''
        return self.c[self.s2i[i0]]

    def ri(self, i0):
        '''
        recurrent state i0?
        '''
        return self.r[self.s2i[i0]]

    def ti(self, i0):
        '''
        transient state i0?
        '''
        return self.t[self.s2i[i0]]

    def ai(self, i0):
        '''
        expected time to be absorbed starting in state i0
        '''
        return self.a[self.s2i[i0]]

    def di(self, i0):
        '''
        relative duration in state i0 starting in starting i0
        '''
        return self.d[self.s2i[i0]]    

    def ii(self, i0):
        '''
        internal state of state i0
        '''
        return self.s2i[i0] 

    def si(self, i0):
        '''
        state name of internal state i0
        '''
        return self.i2s[i0]             

#--------------------------------------------------------------------------------------------------                    

    def info_chain(self):
        print('N:',self.N)        
        print('s:',self.s)
        print('c:',self.c.tolist())
        print('t:',self.t.int().tolist())
        print('r:',self.r.int().tolist())
        print('p:',self.p.tolist())        


#--------------------------------------------------------------------------------------------------            

    def list_classes(self):
        c = self.c
        s = self.s
        S = self.S
        x0 = dict()
        for k0 in range(1+c.max()):
            x0[k0] = [s[i0] for i0 in S[c==k0]]       
        return x0    

    def list_states(self):
        return self.s

    def list_chain(self):
        N = self.N
        P = self.P
        s = self.s
        x0 = [(i0, j0, float(P[i0,j0])) for i0 in range(N) for j0 in range(N) if P[i0,j0]>0] 

        return [(s[i0], s[j0], p0) for (i0,j0,p0) in x0]

#--------------------------------------------------------------------------------------------------                

    def info_class(self, n):
        N = self.N # number of states
        P = self.P # transition probability matrix
        s = self.s # state names
        c = self.c # state classes

        x0 = [(i0, j0, float(P[i0,j0])) for i0 in range(N) for j0 in range(N) if P[i0,j0]>0 and c[i0]==n] 

        return [(s[i0], s[j0], p0) for (i0,j0,p0) in x0]

    def color_class(self, class_id, color_id):
        c = self.c
        c2c = self.c2c

        if 1<=class_id<=c.max() and 0<=color_id<len(MChain.colors):
            c2c[class_id] = color_id

#--------------------------------------------------------------------------------------------------            

    def info_state(self, state, in_states=True, out_states=True):
        N = self.N # number of states
        P = self.P # transition probability matrix
        s = self.s # state names
        s2i = self.s2i # maps state name to state id

        i0 = s2i[state]

        x0 = set()

        if out_states: 
            x0 = x0.union({(i0, j0, float(P[i0,j0])) for j0 in range(N) if P[i0,j0]>0})
        if in_states:
            x0 = x0.union({(j0, i0, float(P[j0,i0])) for j0 in range(N) if P[j0,i0]>0})            

        return [(s[i0], s[j0], p0) for (i0,j0,p0) in x0]


    def rename_state(self, state=None, name=None):
        s = self.s # state names
        s2i = self.s2i # maps state names to state ids
        i2s = self.i2s # maps state ids to state names

        if state in s and name not in s:
            i0 = s2i[state]                   
            s[i0] = name
            i2s[i0] = name
            s2i.pop(state)
            s2i[name] = i0

        return self 


    def add_state(self, state=None, p_list=None):        
        P = self.P # transition probability matrix
        s = self.s # state names
        N = self.N # number of states
        s2i = self.s2i

        if state in s:
            return self
        else:
            s0 = s + [state]
            s2i[state] = N
            P0 = torch.cat((P,torch.zeros((N,1))),1)
            p0 = torch.zeros(N+1)
            if p_list is None:
                p0[N] = 1
            else:
                for x, y in p_list:
                    p0[s2i[x]] = y                    

            P0 = torch.cat((P0,p0[torch.newaxis,:]),0) 
            mc0 = MChain()
            mc0.from_matrix(P0, s0) 

        return mc0 

#--------------------------------------------------------------------------------------------------                   


    def change_transition(self, i0, j0, p0):        
        s = self.s # state names
        S = self.S
        P = self.P # transition probability matrix
        s2i = self.s2i # maps state names to state ids

        p0 = max(0,min(1.0,p0))
        p1 = P[s2i[i0]][s2i[j0]!=S]

        z0 = p1.sum()
        if z0>0:
            P[s2i[i0]][s2i[j0]!=S] = p1*(1-p0)/z0

        P[s2i[i0],s2i[j0]] = p0        

        mc0 = MChain()
        mc0.from_matrix(P, s)

        return mc0         

    def delete_transition(self, i0, j0, absorb=True):
        s = self.s # state names
        P = self.P # transition probability matrix
        s2i = self.s2i # maps state names to state ids

        if (i0 in s) and (j0 in s):
            #P0 = P.copy()
            i1 = s2i[i0]
            j1 = s2i[j0]
            if absorb:
                P[i1,i1] = P[i1,i1] + P[i1,j1]                
            P[i1,j1] = 0           

            mc0 = MChain()
            mc0.from_matrix(P, s)

            return mc0
        else:
            return self


#--------------------------------------------------------------------------------------------------  

    def _initialise(self):
        N = self.N # number of states
        U = self.U # transition probability matrix
        S = self.S # state ids
        s = self.s # state names 

        torch.set_grad_enabled(False)

        u0 = U.sum(axis=1)
        d0 = torch.diagonal(U)
        d0[u0==0] = 1

        U = torch.diagonal_scatter(U,d0)
        self.U = U # unnormalized transition matrix

        P = U/U.sum(axis=1)[:,torch.newaxis]
        self.P = P # probability transition Matrix

        R0 = (P>0).int()
        R = R0
        Rk = R0
        C = torch.zeros((N,N),dtype=torch.int)
        C[:,0] = R0.diagonal()

        for i0 in range(N-1):
            Rk = (0<torch.matmul(Rk,R0)).int()
            R = (0<(R | Rk)).int()
            C[:,i0+1] = Rk.diagonal()*(i0+2)

        self.R = R # reachability matrix 
        self.C = C # circle matrix

        c = (R>torch.t(R)).any(axis=1).int()

        k = 2
        while (0 in c):
            i = S[R[S[c==0][0]]>0]
            c[i] = k
            k = k+1

        c = c - 1  
        self.c = c # class of state

        t = c==0
        self.t = t # transient

        r = c>0
        self.r = r # recurrent


        p = torch.zeros(N, dtype=torch.int)

        for k in range(c.max()):
            l0 = torch.unique(C[c==k+1]) 
            p[c==k+1] = math.gcd(*l0.tolist())    
        self.p = p # period of

        self.c2c = {i:i for i in range(1+c.max())} # maps class to color index         
        self.s2i = {s[i]:i for i in range(N)} # map state name to internal index 
        self.i2s = {i: str(s[i]) for i in range(N)} # map internal index to state name 
        self.i2j = {i0:[j0 for j0 in range(N) if P[i0,j0]>0] for i0 in range(N)} # maps state id to successor ids 
        #self.i2j = {i0:S[P[i0]>0] for i0 in range(N)} # maps state id to successor ids 
        self.i2p = {i0:P[i0][P[i0]>0] for i0 in range(N)} # maps state id to probabilities of successors



    def _process_0(self):        
        N = self.N # number of states 
        S = self.S # state ids
        s = self.s # state names
        P = self.P # transition probability matrix        
        R = self.R # reachability matrix
        c = self.c # state classes
        t = self.t # transient indicator of states
        r = self.r # recurrent indicator of states
        p = self.p # state periodes


        d = torch.zeros(N)

        c0 = torch.unique(c[r])

        for k in c0:
            i0 = S[c==k]
            n0 = i0.shape[0]

            P0 = P[i0,:][:,i0] - torch.eye(n0)
            P0[:,n0-1] = torch.ones(n0)
            A0 = torch.transpose(P0,0,1)

            b0 = torch.zeros(n0)
            b0[n0-1] = 1  

            d[c==k] = self._solve(A0,b0) #torch.linalg.solve(A0,b0)

        self.d = d  #relative duration of stay in state      

        E = -R

        #if i0, j0 recurrent, j0 reachable from i0 ==> visits j0 
        M0 = torch.outer(r,r)
        E[M0] = (E[M0]<0).int()

        #i0 transient, j0 recurrent, j0 reachable from i0, i0 reaches only one class ==> must visits j0
        for i0 in S[t]:  #k, i0 in enumerate(S[t]):
            c0 = E[i0][r]*c[r]
            if 1 == len(torch.unique(c0[c0<0])):
                E[i0][r] = -E[i0][r]    

        P0 = P[t][:,t]
        n0 = P0.shape[0]


        P0 = torch.cat((P0,1-torch.sum(P0,axis=1,keepdims=True)),1)
        P0 = torch.cat((P0,torch.zeros(1,n0+1)),0)

        P0[n0,n0] = 1

        R0 = (P0>0).int()

        s0 = S[t]
        n0 = len(s0)

        x0 = torch.ones(n0+1).bool()

        for j0 in range(n0):
            x0[j0] = False
            R2 = self._getR(R0[x0][:,x0]).bool()
            s1 = s0[x0[:-1]]
            E[s1[~R2[:-1,n0-1]],s0[j0]] = 1
            x0[j0] = True
        self.E = E # visit indicator


        H = torch.zeros((N,N))
        for k in range(c.max()):
            i = c==k+1
            H[torch.outer(i,i)] = 1

        M0 = torch.eye(N)[t]

        for k in range(N):
            b0 = H[:,k]
            b0[k] = 1
            b0 = -torch.matmul(P[t],b0)
            P0 = P[t]
            P0[:,k] = 0
            A0 = (P0 - M0)[:,t]
            H[t,k] = self._solve(A0,b0) #torch.linalg.solve(A0,b0)

        M1 = E!=-1
        H[M1] = E[M1].float()
        self.H = H # H[i,j] probability visiting j starting in i 

        l0 = []
        for k in range(1,1+c.max()):
            l0.append(H[:,[S[k==c][0]]])

        self.Q = torch.hstack(l0)                      

        V = torch.zeros((N,N))

        V[torch.outer(r,r)] = torch.ravel((-E[r][:,r]).float())

        V0 = (E[t][:,r]).float()
        V0[V0!=0] = -1
        V[torch.outer(t,r)] = torch.ravel(V0)   

        H0 = H[t][:,t]
        H0 = H0/(1-torch.diagonal(H0))
        V[torch.outer(t,t)] = torch.ravel(H0)

        self.V = V # V[i,j]: expected visits of j starting in i

        T = -torch.ones((N,N))
        for k in range(N):
            m0 = H[:,k]==1
            if m0.any():
                P0 = P[m0] 
                N0 = P0.shape[0]
                P0[:,k] = 0
                P1 = P0[:,m0]
                P2 = P1 - torch.eye(N0)
                b0 = -torch.ones(N0)
                t0 = self._solve(P2,b0) #torch.linalg.solve(P2,b0)
                T[:,k][m0] = t0 
        self.T = T  # T[i,j] expected time to go from i to j 



    def _process_1(self):           
        N = self.N # number of states
        P = self.P # transition probability matrix        
        t = self.t # transient indicator for states

        P0 = P 
        P0 = P0[t][:,t]               
        n0 = P0.shape[0]

        P0 = torch.cat((P0,1-torch.sum(P0,axis=1,keepdims=True)),1)
        P0 = torch.cat((P0,torch.zeros(1,n0+1)),0)

        P0[n0,n0] = 1

        mc0 = MChain(device=self.device)    

        mc0.from_matrix(P0) 

        mc0._process_0() 

        a = torch.ones(N)
        a[t] = mc0.T[:,-1][:-1]
        self.a = a


    def _show1(self, s, P, c, size=None):
        c2c = self.c2c

        n = len(self.colors)

        if size is None:
            size = '4.5'
        else:
            size = str(size)

        S = torch.arange(len(s))
        G = gv.Digraph()        
        G.attr(rankdir='LR', size=size)

        G.attr('node', shape='doublecircle')

        for k in range(1,1+c.max()):
            color = MChain.colors[c2c[k]%n]
            [G.node(s[i0],style='filled',fillcolor=color) for i0 in S[c==k]]

        G.attr('node', shape='circle')

        [G.node(s[i0]) for i0 in S[c==0]]

        for i0 in S:
            for j0 in S:
                if P[i0,j0]>0:            
                    G.edge(s[i0], s[j0], label='{:.2f}'.format(P[i0,j0].item()))

        display(G)


    def _solve(self, A, b):
        if self.device=='cpu':            
            x = torch.linalg.solve(A,b)
            return x 
        else:
            A0 = A.to(self.device)
            b0 = b.to(self.device)
            x = torch.linalg.solve(A0,b0)
            return x.cpu().detach()    


    def _getR(self, R0):
        N = R0.shape[0]
        if self.device=='cpu':
            R = R0
            for i in range(N-1):
                R = (0<(R + torch.matmul(R,R0))).int()
            return R
        else:
            R1 = R0.float().to(self.device)
            R = R1
            for i in range(N-1):
                R = (0<(R + torch.matmul(R,R1))).float()
            return R.cpu().detach().int()



