import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import matplotlib.patheffects as pe

def plot_mitigation(regression_model: str, threshold:float):

    """
    Plot mitigation effect on PCs

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        Threshold selected, based on R2

    Returns
    ----------
    Plot
    """

    path_cdsf_idle = "../code_6modes/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked = "../code_6modes/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1 = "../code_6modes/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32 = "../code_6modes/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2 = "../code_6modes/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43 = "../code_6modes/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43_train.csv"

    path_dsf_idle = "../code_6modes/03regression_analysis/00standarized/dsf_train_idle.csv"
    path_dsf_parked = "../code_6modes/03regression_analysis/00standarized/dsf_train_parked.csv"
    path_dsf_t1 = "../code_6modes/03regression_analysis/00standarized/dsf_train_t1.csv"
    path_dsf_rpm32 = "../code_6modes/03regression_analysis/00standarized/dsf_train_rpm32.csv"
    path_dsf_t2 = "../code_6modes/03regression_analysis/00standarized/dsf_train_t2.csv"
    path_dsf_rpm43 = "../code_6modes/03regression_analysis/00standarized/dsf_train_rpm43.csv"

    path_scenario_idle = "../code_6modes/03regression_analysis/00standarized/scenario_train_idle.csv"
    path_scenario_parked = "../code_6modes/03regression_analysis/00standarized/scenario_train_parked.csv"
    path_scenario_t1 = "../code_6modes/03regression_analysis/00standarized/scenario_train_t1.csv"
    path_scenario_rpm32 = "../code_6modes/03regression_analysis/00standarized/scenario_train_rpm32.csv"
    path_scenario_t2 = "../code_6modes/03regression_analysis/00standarized/scenario_train_t2.csv"
    path_scenario_rpm43 = "../code_6modes/03regression_analysis/00standarized/scenario_train_rpm43.csv"

    cdsf_idle = pd.read_csv(path_cdsf_idle,index_col=0,header=0,sep=';')
    cdsf_parked = pd.read_csv(path_cdsf_parked,index_col=0,header=0,sep=';')
    cdsf_t1 = pd.read_csv(path_cdsf_t1,index_col=0,header=0,sep=';')
    cdsf_rpm32 = pd.read_csv(path_cdsf_rpm32,index_col=0,header=0,sep=';')
    cdsf_t2 = pd.read_csv(path_cdsf_t2,index_col=0,header=0,sep=';')
    cdsf_rpm43 = pd.read_csv(path_cdsf_rpm43,index_col=0,header=0,sep=';')

    dsf_idle = pd.read_csv(path_dsf_idle,index_col=0,header=0,sep=';')
    dsf_parked = pd.read_csv(path_dsf_parked,index_col=0,header=0,sep=';')
    dsf_t1 = pd.read_csv(path_dsf_t1,index_col=0,header=0,sep=';')
    dsf_rpm32 = pd.read_csv(path_dsf_rpm32,index_col=0,header=0,sep=';')
    dsf_t2 = pd.read_csv(path_dsf_t2,index_col=0,header=0,sep=';')
    dsf_rpm43 = pd.read_csv(path_dsf_rpm43,index_col=0,header=0,sep=';')

    dsf = pd.concat([dsf_idle,dsf_parked],axis=0)
    dsf = pd.concat([dsf, dsf_t1], axis=0)
    dsf = pd.concat([dsf, dsf_rpm32], axis=0)
    dsf = pd.concat([dsf, dsf_t2], axis=0)
    dsf = pd.concat([dsf, dsf_rpm43], axis=0)

    cdsf = pd.concat([cdsf_idle,cdsf_parked],axis=0)
    cdsf = pd.concat([cdsf, cdsf_t1], axis=0)
    cdsf = pd.concat([cdsf, cdsf_rpm32], axis=0)
    cdsf = pd.concat([cdsf, cdsf_t2], axis=0)
    cdsf = pd.concat([cdsf, cdsf_rpm43], axis=0)

    scenario_idle = pd.read_csv(path_scenario_idle,index_col=0,sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked,index_col=0,sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1,index_col=0,sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32,index_col=0,sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2,index_col=0,sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43,index_col=0,sep=';')

    p5  = np.percentile(dsf,5,axis=0)
    p25  = np.percentile(dsf,25,axis=0)
    p50  = np.percentile(dsf,50,axis=0)
    p75  = np.percentile(dsf,75,axis=0)
    p95  = np.percentile(dsf,95,axis=0)

    p5_c = np.percentile(cdsf,5,axis=0)
    p25_c = np.percentile(cdsf,25,axis=0)
    p50_c = np.percentile(cdsf,50,axis=0)
    p75_c = np.percentile(cdsf,75,axis=0)
    p95_c = np.percentile(cdsf,95,axis=0)

    plt.figure()
    s=5

    x=np.linspace(1,720,719)
    N = 5
    a = 0.2 #Alpha dots
    lw = 1 #Line width

    print(dsf.shape)
    print(p5 .shape)

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.scatter(x,p5 ,color='red',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p5 ,N),c='black',linestyle='dotted',lw=lw,path_effects=[pe.Stroke(linewidth=lw+2, foreground='red'), pe.Normal()])

    plt.scatter(x,p25 ,color='red',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p25 ,N),c='black',linestyle='dashdot',lw=lw,path_effects=[pe.Stroke(linewidth=lw+3, foreground='red'), pe.Normal()])

    plt.scatter(x,p50 ,color='red',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p50 ,N),c='black',lw=lw,path_effects=[pe.Stroke(linewidth=lw+3, foreground='red'), pe.Normal()])

    plt.scatter(x,p75 ,color='red',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p75 ,N),c='black',linestyle='dashdot',lw=lw,path_effects=[pe.Stroke(linewidth=lw+3, foreground='red'), pe.Normal()])

    plt.scatter(x,p95 ,color='red',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p95 ,N),c='black',linestyle='dotted',lw=lw,path_effects=[pe.Stroke(linewidth=lw+2, foreground='red'), pe.Normal()])

    #CORRECTED
    plt.scatter(x,p5_c,color='green',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p5_c,N),c='black',linestyle='dotted',lw=lw,path_effects=[pe.Stroke(linewidth=lw+2, foreground='green'), pe.Normal()])

    plt.scatter(x,p25_c,color='green',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p25_c,N),c='black',linestyle='dashdot',lw=lw,path_effects=[pe.Stroke(linewidth=lw+3, foreground='green'), pe.Normal()])

    plt.scatter(x,p50_c,color='green',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p50_c,N),c='black',lw=lw,path_effects=[pe.Stroke(linewidth=lw+3, foreground='green'), pe.Normal()])

    plt.scatter(x,p75_c,color='green',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p75_c,N),c='black',linestyle='dashdot',lw=lw,path_effects=[pe.Stroke(linewidth=lw+3, foreground='green'), pe.Normal()])

    plt.scatter(x,p95_c,color='green',s=s,alpha=a)
    plt.semilogx(x,uniform_filter1d(p95_c,N),c='black',linestyle='dotted',lw=lw,path_effects=[pe.Stroke(linewidth=lw+2, foreground='green'), pe.Normal()])

    plt.xlim([1,720])
    plt.ylim([-2,2])
    plt.yticks([-2,-1,0,1,2])
    # plt.xticks([])

    print(dsf_idle.shape)

    plt.show()

# plot_mitigation('svr', 0.0)
plot_mitigation('svr', 0.6)
# plot_mitigation('knn', 0.6)

# plot_mitigation('adaptive_order1', 0.0)
# plot_mitigation('adaptive_order1', 0.3)
# plot_mitigation('adaptive_order1', 0.6)

# plot_mitigation('adaptive_order2', 0.0)
# plot_mitigation('adaptive_order2', 0.3)
# plot_mitigation('adaptive_order2', 0.6)

# plot_mitigation('adaptive_order3', 0.0)
# plot_mitigation('adaptive_order3', 0.3)
# plot_mitigation('adaptive_order3', 0.6)

# plot_mitigation('adaptive_order4', 0.0)
# plot_mitigation('adaptive_order4', 0.3)
# plot_mitigation('adaptive_order4', 0.6)

# plot_mitigation('adaptive_order5', 0.0)
# plot_mitigation('adaptive_order5', 0.3)
# plot_mitigation('adaptive_order5', 0.6)

# plot_mitigation('adaptive_order6', 0.0)
# plot_mitigation('adaptive_order6', 0.3)
# plot_mitigation('adaptive_order6', 0.6)