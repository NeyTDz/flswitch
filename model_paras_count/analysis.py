import math
import numpy as np
import matplotlib.pyplot as plt



marker_list = ['o','^','s','p','x']
#marker_list = marker_list[:len(model_names)]
aplot_list = []
asc_list = []
#plt.xlabel("Error Distance(m)", fontdict={'family' : 'Times New Roman', 'size':12})
#plt.ylabel("CDF", fontdict={'family' : 'Times New Roman', 'size':12})

def plot(costs,batches,sparse):

    top = np.arange(costs.shape[1]) + 1
    for i in range(len(costs)):
        cost, batch = costs[i], batches[i]
        aplot = plt.plot(top,cost,linewidth = 1,marker = "o",markersize=4,zorder=1)
        #asc = plt.scatter(top,cost,edgecolors=aplot[0].get_color(),markersize=8)
    plt.title("sparse={}".format(sparse))
    plt.xlabel("top")
    plt.ylabel("cost")
    plt.legend(labels = ["batch="+str(batch) for batch in batches], fontsize=7)
    #plt.ylim(20,140)
    plt.savefig("./pics/sparse{}.png".format(sparse))
    plt.show()

'''
for i,acc in enumerate(accs):

    aplot = plt.plot(error_dist,acc,linewidth = 1,zorder=1)
    points = np.arange(0,len(error_dist),10)
    points = np.append(points,[len(error_dist)-1])
    points_err_dist = error_dist[points]
    points_acc = acc[points]
    color = aplot[0].get_color()
    if marker_list[i] != 'x':
        asc = plt.scatter(points_err_dist,points_acc,edgecolors=color, c = 'w',marker = marker_list[i],zorder=3)
    else:
        asc = plt.scatter(points_err_dist,points_acc,c = color,marker = marker_list[i],zorder=2)
    aplot_list.append(aplot)
    asc_list.append(asc)

plt.legend(labels = [model for model in model_names])
plt.savefig("./pics/compare.png")
plt.show()    

'''