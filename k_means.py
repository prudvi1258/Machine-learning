
import pandas as pand
import xlrd as xl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rd
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('result.pdf')

def dst(x1,y1,x2,y2):
    eq_dist=np.sqrt(((x1-(x2))**2)+((y1-(y2))**2))
    return eq_dist

def init_centroids(datasetf,k):
    centers=[]
    #z=rd.randint(0,len(df))
    z=0
    print(z)
    centers.append((datasetf.iloc[0,0],datasetf.iloc[0,1]))
    for i in range(1,k):
        max_dist=0;
        max_dist_index=0;
        for row in datasetf.itertuples():
            distances_list=[]
            m=row[0]
            x=row[1]
            y=row[2]
            for j in range(i):
                (cx,cy)=centers[j]
                distance=dst(cx,cy,x,y)
                distances_list.append(distance)
            distances_list_np=np.array(distances_list)
            min_dist=np.min(distances_list_np)
            if min_dist>max_dist:
                max_dist=min_dist
                max_dist_index=m
        centers.append((datasetf.iloc[max_dist_index,0],datasetf.iloc[max_dist_index,1]))
    return centers

def avg_error(centers_np,clusters,datasetf,k):
    avg_dis=[]
    for i in range(k):
        avg_dis.append([])
    for h in range(k):
        (cx,cy)=centers_np[h]
        for i in clusters[h]:
            x=datasetf.iloc[i,0]
            y=datasetf.iloc[i,1]
            distance=dst(cx,cy,x,y)
            avg_dis[h].append(distance)
        avg_dis[h]=np.mean(avg_dis[h])
    return np.mean(avg_dis)

def k_means(datasetf,k):
    centers=init_centroids(datasetf,k)
    #print(centers)
    centers_np=np.array(centers)
    centers_prev=np.zeros(centers_np.shape)
    #centers_prev=centers_np
    while  not np.array_equal(centers_np,centers_prev):
        centers_prev=centers_np
        clusters=[]
        #print(centers_np)
        for q in range(k):
            clusters.append([])
        for row in datasetf.itertuples():
            i=row[0]
            x=row[1]
            y=row[2]
            (cx,cy)=centers_np[0]
            min_dist=dst(cx,cy,x,y)
            cluster_index=0
            for h,center in enumerate(centers_np):
                (cx,cy)=center
                distance=dst(cx,cy,x,y)
                if(distance<min_dist):
                    min_dist=distance
                    cluster_index=h
            clusters[cluster_index].append(i)
        centers_np=[]
        for h in range(k):
            centers_np.append((datasetf.iloc[clusters[h],0].mean(),datasetf.iloc[clusters[h],1].mean()))

    #print(clusters[2])
    #colors = (i + j for j in 'o<.' for i in 'bgrcmyk')
    #colors=['r','b','g','c']
    colors = cm.rainbow(np.linspace(0, 1, k))
    for h in range(k):
        datasetf_slice=datasetf.iloc[clusters[h],:]
        plt.scatter(datasetf_slice[0],datasetf_slice[1],c=colors[h])
    pp.savefig()
    plt.show()
    return avg_error(centers_np,clusters,datasetf,k)



datasetf = pand.read_excel('dataset.xlsx',header=None)
#k_means_2(df);
error1=k_means(datasetf,1)
error2=k_means(datasetf,2)
error3=k_means(datasetf,3)
error4=k_means(datasetf,4)
error5=k_means(datasetf,5)
error=[error1,error2,error3,error4,error5]
plt.plot([1,2,3,4,5],error)
pp.savefig()
plt.show()
pp.close()