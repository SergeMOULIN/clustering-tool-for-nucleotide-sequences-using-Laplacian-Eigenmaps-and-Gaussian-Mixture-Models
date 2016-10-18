#-*-coding:utf8-*-

import numpy as np
import random as rd
from sklearn.mixture import GMM
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mt
from __future__ import print_function


############################################
##   Importing the distance matrix :
############################################

nc = ['NC_020374', 'NC_020153', 'NC_014768', 'NC_017615', 'NC_017613', 'NC_013844', 'NC_012896', 'NC_012894', 'NC_004826', 'NC_000928', 'NC_011121', 'NC_009938', 'NC_009460', 'NC_009462', 'NC_011122', 'NC_011037', 'NC_009463', 'NC_009461', 'NC_008945', 'NC_008075', 'NC_002767', 'NC_004022', 'NC_002547', 'NC_018602', 'NC_018596', 'NC_018597', 'NC_019571', 'NC_018363', 'NC_019803', 'NC_019809', 'NC_019810', 'NC_016199', 'NC_016172', 'NC_014486', 'NC_016197', 'NC_016186', 'NC_016198', 'NC_016128', 'NC_016019', 'NC_015927', 'NC_015925', 'NC_014870', 'NC_017747', 'NC_017750', 'NC_016200', 'NC_016129', 'NC_016127', 'NC_015924', 'NC_015245', 'NC_014181', 'NC_013818', 'NC_013831', 'NC_013827', 'NC_013817', 'NC_013808', 'NC_013253', 'NC_013067', 'NC_013065', 'NC_014282', 'NC_013848', 'NC_013824', 'NC_013821', 'NC_013815', 'NC_013813', 'NC_013807', 'NC_008047', 'NC_007934', 'NC_005928', 'NC_003416', 'NC_004806', 'NC_001328', 'NC_005941', 'NC_005305', 'NC_001861', 'NC_001327', 'NC_012308', 'NC_012309', 'NC_010383', 'NC_010773', 'NC_009885', 'NC_011300', 'NC_010690', 'NC_010527', 'NC_008693', 'NC_008534', 'NC_008231', 'NC_008692', 'NC_004298', 'NC_003415', 'NC_002681', 'NC_002354', 'NC_011127', 'NC_012147', 'NC_009680', 'NC_008067', 'NC_002544', 'NC_002546', 'NC_002529', 'NC_008074', 'NC_002545']
## nc est la liste des références des espèces considérées


w = []

for fichier in ['matriceDistances_ND3_muscle.txt']:
    fic = open(fichier).read().split('\n')[0:100]
    for ligne in fic:
        w.append(np.array(ligne.split()[0:100],float))
    w=np.asarray(w)

# w[i,j] est la distance inter-génomique entre l'espèce nc[i] et l'espèce nc[j].


################################################################################################################
##  Matrices de distances pour les Nematodes d'une part, pour les Platyhelminthes d'autre part (un peu inutile):
################################################################################################################

## m'a néanmoins permis de tester de manière pratique si les clusters dissociaient bien nos 2 familles d'espèces,
## cela dit les arbres phylo accompagnés des taxonomies sont suffisants.

ncPlatyNotSorted=['NC_002529','NC_009463','NC_008945','NC_017615','NC_017613','NC_011037','NC_002547','NC_020153','NC_009460','NC_008075','NC_011121','NC_011122','NC_000928','NC_009462','NC_009461','NC_020374','NC_012896','NC_004826','NC_009938','NC_004022','NC_012894','NC_002767','NC_013844','NC_014768','NC_011127','NC_012147','NC_002354','NC_002546','NC_002544','NC_008074','NC_002545','NC_008067','NC_009680']

ncPlaty=[]
ncNema=[]
for u in nc :
    if u in ncPlatyNotSorted :
        ncPlaty.append(u)
    else :
        ncNema.append(u)

# ncPlaty et ncNema sont nos listes de références

lPlaty=[]
lNema=[]
for (u,v) in enumerate(nc) :
    if v in ncPlaty :
        lPlaty.append(u)
    else :
        lNema.append(u)

wPlaty=np.copy(w)
a=0
for i in lNema :
    wPlaty = np.delete(wPlaty, (i-a), axis=0)
    wPlaty = np.delete(wPlaty, (i-a), axis=1)
    a+=1

wNema=np.copy(w)
a=0
for i in lPlaty :
    wNema = np.delete(wNema, (i-a), axis=0)
    wNema = np.delete(wNema, (i-a), axis=1)
    a+=1

# wPlaty et wNema sont nos matrices de distances (pour un "sous-clustering", par exemple)


#################################
##  Définition de nos fonctions :
#################################




def plotembedding(vecprop,p) :  ## sert à modéliser nos clustering en 2D (projection de nos vecteurs-espèces sur 2 vecteurs propres)
    ## vecprop est une la liste des vecteurs propres retenus (2,3 ou 4)
    ## p est la liste de vecteurs-espèces associés à un cluster particulier
    coord=[]
    coord.append(vecprop[0,:])
    coord.append(vecprop[1,:])
    
    if len(vecprop)>2:
        coord.append(vecprop[2,:])
    
    if len(vecprop)==4 :
        coord.append(vecprop[3,:])
    
    dicoCoul={0:'r',1:'b',2:'y',3:'c',4:'k',5:'m',6:'g',7:'c',8:'springgreen',9:'brown'}
    listCoul = []
    for i in range(len(p)):
        listCoul.append(dicoCoul[p[i]])
    scatter(coord[0],coord[1],color=listCoul) 
    xlabel('Component 1')
    ylabel('Component 2')
    title('GMM clustering with 2 eigenvectors')
    show()
    
    if len(vecprop)>2 :
        scatter(coord[0],coord[2],color=listCoul) 
        xlabel('Component 1')
        ylabel('Component 3')
        title('GMM clustering with 2 eigenvectors')
        show()
        scatter(coord[1],coord[2],color=listCoul) 
        xlabel('Component 2')
        ylabel('Component 3')
        title('GMM clustering with 2 eigenvectors')
        show()
    
    if len(vecprop)==4 :
        scatter(coord[0],coord[3],color=listCoul) 
        xlabel('Component 1')
        ylabel('Component 4')
        title('GMM clustering with 2 eigenvectors')
        show()
        scatter(coord[1],coord[3],color=listCoul) 
        xlabel('Component 2')
        ylabel('Component 4')
        title('GMM clustering with 2 eigenvectors')
        show()
        scatter(coord[2],coord[3],color=listCoul) 
        xlabel('Component 3')
        ylabel('Component 4')
        title('GMM clustering with 2 eigenvectors')
        show()


def scaledimage(W): ## représentation graphique de notre matrice de similarité W
    (N, M) = W.shape 
    fig,ax=plt.subplots()
    exts = (0, M, 0, N)
    ax.imshow(W, interpolation='nearest', cmap=cm.autumn_r, extent=exts)
    cbar = fig.colorbar(ax.imshow(W, interpolation='nearest', cmap=cm.autumn_r, extent=exts), ticks=[0,1])
    cbar.ax.set_yticklabels(['0','1'])
    ax.xaxis.set_major_locator(mt.NullLocator())
    ax.yaxis.set_major_locator(mt.NullLocator())
    return ax


def Lap_eigen_rw(w) :  ## Lap_eigen_rw(w) est notre laplacienne normalisée, avec w une matrice de similarité
    d=np.zeros((w.shape[0],w.shape[0]))
    for i in range(w.shape[0]) :
        for j in range(w.shape[0]) :
            d[i,i]=d[i,i]+w[i,j]        # d = matrice des degrés
    return(dot(np.linalg.inv(d),d-w))
    

#################################################################################################
## FONCTION MERE : gclust(m,refs,simil=False,nbClusters='BIC',drawgraphs=True,dim_embedding=0.01)
#################################################################################################

# Serge: Ok, merci pour les commentaires.
# Serge: Je les ai placé juste après la définition de la fonction. C'est là qu'on les place normalement. 
# Serge : J'ai changé "dim_embedding" par "delta" qui est plus raccord avec ce que j'écris dans l'article. On peut 
# appeller ça différement mais dim_embedding me semplbe embigu, on dirait qu'il s'agit du nombre de vecteurs proores retenus.

def gclust(m,refs,simil=False,nbClusters='BIC',drawgraphs=True, delta=0.01) :
    '''
    -  m : matrice de distances inter-génomiques si simil=False
           Si simil=True, m est une matrice de similitude
    -  refs : liste des noms des espèces étudiées 
          (m[i,j]=m[j,i]=distance entre espèce i et espèce j, i.e. entre refs[i] et refs[j])
    -  nbClusters : nombre de clusters à considérer pour la classification (critère BIC par défaut)
    -  drawgraphs=True : dessine les grahiques
    -  dim_embedding : critère de sélection du nombre de vecteurs propres à considérer
    '''
    N=m.shape[0]  ## N = dimension de notre matrice m (nombre de séquences considérées)
    
    w=np.copy(m)
    
    if not simil :
        w=1-w/w.max()  ## w est la matrice de similarité issue de m
    
    if drawgraphs==True :
        scaledimage(w)
        plt.show()
    
    Lrw=Lap_eigen_rw(w)  ## Lrw est la laplacienne normalisée
    
    vp=[]
    b=np.linalg.eig(Lrw)   
    for i in range(N) :
        vp.append((b[0][i],b[1].T[i]))

    VP=sorted(vp)    ##VP est la liste des valeurs propres et vecteurs propres triés par ordre croissant selon les valeurs propres
    
    vaP= []
    for i in range(N) :
        vaP.append(VP[i][0])  ## vaP est la liste des valeurs propres uniquement (par ordre croissant)
    
   
    title('Values of the 10 smallest eigenvalues')
    plot(range(11),vaP[0:11])    
    show()
    
    i=0
    while (VP[i+1][0]>VP[i][0]+delta):
        i+=1
    
    vePclasses=[]
    for j in range(N) :
        vePclasses.append(VP[j][1])
    
    
    vecprop=np.asarray(vePclasses[1:i+1])
   
    print("The ", i, " first eigenvalues were selected.")
    
    if nbClusters=='BIC' :
        H1 = []
        for i in range(10) :
            Xaxes = range(1,10)
            for k in Xaxes:
                gmm = GMM(n_components=k, covariance_type='full')
                h = gmm.fit(vecprop.T)
                H1.append(h.aic(vecprop.T))
            #plot(Xaxes,H1[10*i:10*i+9],label='full')
            #title('Bayesian Information Criterion of the Gaussian Mixture Models \n') 
            #show()    
        H2=sorted(H1)
        k=(H1.index(H2[0])%9)+1       #  k = Nombre de clusters pour une classification optimale
        
        
    else :
        k=nbClusters
    
    Bic = float('inf')
    gmm = GMM(n_components=k, covariance_type='full')
    for i in range(20):
        h0 = gmm.fit(vecprop.T)
        B =  h0.bic(vecprop.T)
        if B < Bic:
            Bic = B
            h = h0

    p0 = h.predict_proba(vecprop.T)

    p = np.zeros(len(p0))
    for i in range(len(p0)):
        p[i] = argmax (p0[i])
    
    groupe=[]
    for i in range(k) :
        groupe.append([])
        
    for i in range(N) :
        groupe[int(p[i])].append(refs[i])
    
    if drawgraphs==True :
        plotembedding(vecprop,p)
    
    return groupe


#########################################################################################
##     Transformation de notre database en .fasta, avec les séquences alignées, une 
##  partie de la taxonomie des espèces, et le numéro de cluster qui leur est associé (de 
##  manière à pouvoir les représenter sur un arbre phylogénétique par exemple) 
#########################################################################################


groupe=gclust(w,nc)



'''
Fonction qui permet de savoir à quel cluster appartient une espèce donnée.
ref : nom d'une espèce donnée "NC_..."
groupe : résultat d'un gclust, c'est-à-dire une liste de listes, chaque sous-liste 
contenant toute les espèces contenues dans un cluster.
'''

def numClust(ref,groupe) :
    a=0
    while not ref in groupe[a] :
           a+=1
    return(a)



for fichier in ['muscle_ND3.txt']:
    fic=open(fichier).read().split("\n")

fic.remove('')

Y=[]
for i in range(100):
    Y.append((fic[2*i],fic[2*i+1]))

dicoMuscle={}
for (u,v) in Y:
    dicoMuscle[u]=v
    

#################################

###    ARBRES 



##refs especes

from Bio.Align.Applications import *
from Bio.Align.AlignInfo import *
from Bio import AlignIO
from Bio import Phylo
from Bio import SeqIO
import numpy as np
from Bio.SeqRecord import SeqRecord

from Bio import Entrez
import urllib2
Entrez.email="christophe.guyeux@univ-fcomte.fr"

def espece(ref):
    nom = Entrez.efetch(db="nuccore", id=ref, rettype="gbk", retmode="text").read().split('taxname "')[1].split('"')[0]
    lineage = Entrez.efetch(db="nuccore", id=ref, rettype="gbk", retmode="text").read().split('lineage "')[1].split('"')[0].replace('\n','').split('; ') 
    return(nom,lineage)




# Ligne de code qui permet d'associer le numéro de cluster et une patie de la taxonomie à chaque espèce.  
seqRecord=[SeqRecord(Seq(Y[k][1]),id=Y[k][0],description=str(espece(Y[k][0])[1][2:6])+'--> cluster '+str(numClust(Y[k][0],groupe))) for k in range(100)]



for u in seqRecord :
    print u.format('fasta')

output_handle = open("muscle_10clusters.fasta", "w")
SeqIO.write(seqRecord, output_handle, "fasta")
output_handle.close()
