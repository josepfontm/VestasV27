import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,GridSearchCV
import math

def decision_tree(threshold:float, plot_modes:list, plot_regression = False):
    
    """
    Perform support vector regresion.
    
    Parameters
    ----------
    threshold: float
        DSFs are only corrected if the R^2 is higher than the threshold defined.
        If DSFs has high correlation with multiple PCs (Env and Op) the highest is used to correct DSF.

    plot_modes: list
        Show correlation plots for these modes.

    plot_regression: bool
        Show which DSFs should be corrected, based on the R2 obtained and the threshold selected.

    Returns
    ----------
    corrected_dsf: array
        Array with corrected dsf's based on environmental and operational data.
    """

    #Change paths if necessary
    path_scenario_idle = '../code_6modes/03regression_analysis/00standarized/scenario_idle.csv'
    path_scenario_parked = '../code_6modes/03regression_analysis/00standarized/scenario_parked.csv'
    path_scenario_t1 = '../code_6modes/03regression_analysis/00standarized/scenario_t1.csv'
    path_scenario_rpm32 = '../code_6modes/03regression_analysis/00standarized/scenario_rpm32.csv'
    path_scenario_t2 = '../code_6modes/03regression_analysis/00standarized/scenario_t2.csv'
    path_scenario_rpm43 = '../code_6modes/03regression_analysis/00standarized/scenario_rpm43.csv'

    path_scenario_train_idle = '../code_6modes/03regression_analysis/00standarized/scenario_train_idle.csv'
    path_scenario_train_parked = '../code_6modes/03regression_analysis/00standarized/scenario_train_parked.csv'
    path_scenario_train_t1 = '../code_6modes/03regression_analysis/00standarized/scenario_train_t1.csv'
    path_scenario_train_rpm32 = '../code_6modes/03regression_analysis/00standarized/scenario_train_rpm32.csv'
    path_scenario_train_t2 = '../code_6modes/03regression_analysis/00standarized/scenario_train_t2.csv'
    path_scenario_train_rpm43 = '../code_6modes/03regression_analysis/00standarized/scenario_train_rpm43.csv'

    scenario_idle = pd.read_csv(path_scenario_idle, index_col=0,sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked, index_col=0,sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1, index_col=0,sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32, index_col=0,sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2, index_col=0,sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43, index_col=0,sep=';')

    scenario_train_idle = pd.read_csv(path_scenario_train_idle, index_col=0,sep=';')
    scenario_train_parked = pd.read_csv(path_scenario_train_parked, index_col=0,sep=';')
    scenario_train_t1 = pd.read_csv(path_scenario_train_t1, index_col=0,sep=';')
    scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32, index_col=0,sep=';')
    scenario_train_t2 = pd.read_csv(path_scenario_train_t2, index_col=0,sep=';')
    scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43, index_col=0,sep=';')

    #Load paths for pc's environemntal data
    path_pca_env_idle = '../code_6modes/03regression_analysis/00standarized/pca_env_idle.csv'
    path_pca_env_parked = '../code_6modes/03regression_analysis/00standarized/pca_env_parked.csv'
    path_pca_env_t1 = '../code_6modes/03regression_analysis/00standarized/pca_env_t1.csv'
    path_pca_env_rpm32 = '../code_6modes/03regression_analysis/00standarized/pca_env_rpm32.csv'
    path_pca_env_t2 = '../code_6modes/03regression_analysis/00standarized/pca_env_t2.csv'
    path_pca_env_rpm43 = '../code_6modes/03regression_analysis/00standarized/pca_env_rpm43.csv'

    pca_env_idle = pd.read_csv(path_pca_env_idle,index_col=0,sep=';')
    pca_env_parked = pd.read_csv(path_pca_env_parked,index_col=0,sep=';')
    pca_env_t1 = pd.read_csv(path_pca_env_t1,index_col=0,sep=';')
    pca_env_rpm32 = pd.read_csv(path_pca_env_rpm32,index_col=0,sep=';')
    pca_env_t2 = pd.read_csv(path_pca_env_t2,index_col=0,sep=';')
    pca_env_rpm43 = pd.read_csv(path_pca_env_rpm43,index_col=0,sep=';')

    path_pca_env_train_idle = '../code_6modes/03regression_analysis/00standarized/pca_env_train_idle.csv'
    path_pca_env_train_parked = '../code_6modes/03regression_analysis/00standarized/pca_env_train_parked.csv'
    path_pca_env_train_t1 = '../code_6modes/03regression_analysis/00standarized/pca_env_train_t1.csv'
    path_pca_env_train_rpm32 = '../code_6modes/03regression_analysis/00standarized/pca_env_train_rpm32.csv'
    path_pca_env_train_t2 = '../code_6modes/03regression_analysis/00standarized/pca_env_train_t2.csv'
    path_pca_env_train_rpm43 = '../code_6modes/03regression_analysis/00standarized/pca_env_train_rpm43.csv'

    pca_env_train_idle = pd.read_csv(path_pca_env_train_idle,index_col=0,sep=';')
    pca_env_train_parked = pd.read_csv(path_pca_env_train_parked,index_col=0,sep=';')
    pca_env_train_t1 = pd.read_csv(path_pca_env_train_t1,index_col=0,sep=';')
    pca_env_train_rpm32 = pd.read_csv(path_pca_env_train_rpm32,index_col=0,sep=';')
    pca_env_train_t2 = pd.read_csv(path_pca_env_train_t2,index_col=0,sep=';')
    pca_env_train_rpm43 = pd.read_csv(path_pca_env_train_rpm43,index_col=0,sep=';')

    #Load paths for pc's operational data
    path_pca_op_idle = '../code_6modes/03regression_analysis/00standarized/pca_op_idle.csv'
    path_pca_op_parked = '../code_6modes/03regression_analysis/00standarized/pca_op_parked.csv'
    path_pca_op_t1 = '../code_6modes/03regression_analysis/00standarized/pca_op_t1.csv'
    path_pca_op_rpm32 = '../code_6modes/03regression_analysis/00standarized/pca_op_rpm32.csv'
    path_pca_op_t2 = '../code_6modes/03regression_analysis/00standarized/pca_op_t2.csv'
    path_pca_op_rpm43 = '../code_6modes/03regression_analysis/00standarized/pca_op_rpm43.csv'

    path_pca_op_train_idle = '../code_6modes/03regression_analysis/00standarized/pca_op_train_idle.csv'
    path_pca_op_train_parked = '../code_6modes/03regression_analysis/00standarized/pca_op_train_parked.csv'
    path_pca_op_train_t1 = '../code_6modes/03regression_analysis/00standarized/pca_op_train_t1.csv'
    path_pca_op_train_rpm32 = '../code_6modes/03regression_analysis/00standarized/pca_op_train_rpm32.csv'
    path_pca_op_train_t2 = '../code_6modes/03regression_analysis/00standarized/pca_op_train_t2.csv'
    path_pca_op_train_rpm43 = '../code_6modes/03regression_analysis/00standarized/pca_op_train_rpm43.csv'

    pca_op_idle = pd.read_csv(path_pca_op_idle,index_col=0,sep=';')
    pca_op_parked = pd.read_csv(path_pca_op_parked,index_col=0,sep=';')
    pca_op_t1 = pd.read_csv(path_pca_op_t1,index_col=0,sep=';')
    pca_op_rpm32 = pd.read_csv(path_pca_op_rpm32,index_col=0,sep=';')
    pca_op_t2 = pd.read_csv(path_pca_op_t2,index_col=0,sep=';')
    pca_op_rpm43 = pd.read_csv(path_pca_op_rpm43,index_col=0,sep=';')

    pca_op_train_idle = pd.read_csv(path_pca_op_train_idle,index_col=0,sep=';')
    pca_op_train_parked = pd.read_csv(path_pca_op_train_parked,index_col=0,sep=';')
    pca_op_train_t1 = pd.read_csv(path_pca_op_train_t1,index_col=0,sep=';')
    pca_op_train_rpm32 = pd.read_csv(path_pca_op_train_rpm32,index_col=0,sep=';')
    pca_op_train_t2 = pd.read_csv(path_pca_op_train_t2,index_col=0,sep=';')
    pca_op_train_rpm43 = pd.read_csv(path_pca_op_train_rpm43,index_col=0,sep=';') 

    #Merge PCs environmental and operational data
    pca_idle = pd.concat([pca_env_idle, pca_op_idle], axis=1)
    pca_parked = pd.concat([pca_env_parked, pca_op_parked], axis=1)
    pca_t1 = pd.concat([pca_env_t1, pca_op_t1], axis=1)
    pca_rpm32 = pd.concat([pca_env_rpm32, pca_op_rpm32], axis=1)
    pca_t2 = pd.concat([pca_env_t2,pca_op_t2],axis=1)
    pca_rpm43 = pd.concat([pca_env_rpm43, pca_op_rpm43], axis = 1)

    pca_train_idle = pd.concat([pca_env_train_idle,pca_op_train_idle],axis=1)
    pca_train_parked = pd.concat([pca_env_train_parked,pca_op_train_parked],axis=1)
    pca_train_t1 = pd.concat([pca_env_train_t1,pca_op_train_t1],axis=1)
    pca_train_rpm32 = pd.concat([pca_env_train_rpm32,pca_op_train_rpm32],axis=1)
    pca_train_t2 = pd.concat([pca_env_train_t2,pca_op_train_t2],axis=1)
    pca_train_rpm43 = pd.concat([pca_env_train_rpm43,pca_op_train_rpm43],axis=1)

    #Paths damage-sensitive features
    path_dsf_idle = '../code_6modes/03regression_analysis/00standarized/dsf_idle.csv'
    path_dsf_parked = '../code_6modes/03regression_analysis/00standarized/dsf_parked.csv'
    path_dsf_t1 = '../code_6modes/03regression_analysis/00standarized/dsf_t1.csv'
    path_dsf_rpm32 = '../code_6modes/03regression_analysis/00standarized/dsf_rpm32.csv'
    path_dsf_t2 = '../code_6modes/03regression_analysis/00standarized/dsf_t2.csv'
    path_dsf_rpm43 = '../code_6modes/03regression_analysis/00standarized/dsf_rpm43.csv'

    dsf_idle = pd.read_csv(path_dsf_idle,index_col=0,sep=';')
    dsf_parked = pd.read_csv(path_dsf_parked,index_col=0,sep=';')
    dsf_t1 = pd.read_csv(path_dsf_t1,index_col=0,sep=';')
    dsf_rpm32 = pd.read_csv(path_dsf_rpm32,index_col=0,sep=';')
    dsf_t2 = pd.read_csv(path_dsf_t2,index_col=0,sep=';')
    dsf_rpm43 = pd.read_csv(path_dsf_rpm43,index_col=0,sep=';')

    path_dsf_train_idle = '../code_6modes/03regression_analysis/00standarized/dsf_train_idle.csv'
    path_dsf_train_parked = '../code_6modes/03regression_analysis/00standarized/dsf_train_parked.csv'
    path_dsf_train_t1 = '../code_6modes/03regression_analysis/00standarized/dsf_train_t1.csv'
    path_dsf_train_rpm32 = '../code_6modes/03regression_analysis/00standarized/dsf_train_rpm32.csv'
    path_dsf_train_t2 = '../code_6modes/03regression_analysis/00standarized/dsf_train_t2.csv'
    path_dsf_train_rpm43 = '../code_6modes/03regression_analysis/00standarized/dsf_train_rpm43.csv'

    dsf_train_idle = pd.read_csv(path_dsf_train_idle,index_col=0,sep=';')
    dsf_train_parked = pd.read_csv(path_dsf_train_parked,index_col=0,sep=';')
    dsf_train_t1 = pd.read_csv(path_dsf_train_t1,index_col=0,sep=';')
    dsf_train_rpm32 = pd.read_csv(path_dsf_train_rpm32,index_col=0,sep=';')
    dsf_train_t2 = pd.read_csv(path_dsf_train_t2,index_col=0,sep=';')
    dsf_train_rpm43 = pd.read_csv(path_dsf_train_rpm43,index_col=0,sep=';')

    #Result matrices
    r2_idle = np.zeros(np.shape(dsf_idle)[1])
    r2_parked = np.zeros(np.shape(dsf_parked)[1])
    r2_t1 = np.zeros(np.shape(dsf_t1)[1])
    r2_rpm32 = np.zeros(np.shape(dsf_rpm32)[1])
    r2_t2 = np.zeros(np.shape(dsf_t2)[1])
    r2_rpm43 = np.zeros(np.shape(dsf_rpm43)[1])

    r2_modes = [r2_idle, r2_parked, r2_t1, r2_rpm32, r2_t2, r2_rpm43]
    dsf_modes = [dsf_idle, dsf_parked, dsf_t1, dsf_rpm32, dsf_t2, dsf_rpm43]
    dsf_train_modes = [dsf_train_idle, dsf_train_parked, dsf_train_t1, dsf_train_rpm32, dsf_train_t2, dsf_train_rpm43]
    pca_modes = [pca_idle, pca_parked, pca_t1, pca_rpm32, pca_t2, pca_rpm43]
    pca_train_modes = [pca_train_idle, pca_train_parked, pca_train_t1, pca_train_rpm32,pca_train_t2, pca_train_rpm43]
    file_modes = ['idle','parked','t1','rpm32','t2','rpm43']
    scenarios = [scenario_idle,scenario_parked,scenario_t1,scenario_rpm32,scenario_t2,scenario_rpm43]


    colors = {'undamaged':'green',
              '15cm':'yellow',
              '30cm':'orange',
              '45cm':'red',
              'repaired':'blue'}
    
    s = 3 #Size dots scatter

    for r2_mode,dsf_mode,dsf_train_mode,pca_mode,pca_train_mode,file_mode,scenario in zip(r2_modes,dsf_modes,dsf_train_modes,pca_modes,pca_train_modes,file_modes,scenarios):
        tree = DecisionTreeRegressor()

        observations = pca_train_mode.to_numpy().shape[0]

        #Parameters GridSearch CV  
        param_grid = {'max_depth': [2,4,6,8,10,20],
                      'splitter': ['best','random'],
                      'min_samples_leaf':[3,5,10,20,30],
                      'ccp_alpha':[0.001,0.01,0.1]}
                      
        for pc in range(np.shape(dsf_mode)[1]):
            r2 = []
            x = pca_mode.to_numpy()

            y = dsf_mode.iloc[:,pc].to_numpy()
            y = y.ravel()

            #Training data (Repaired)
            x_train = pca_train_mode.to_numpy()

            y_train = dsf_train_mode.iloc[:,pc].to_numpy()
            y_train = y_train.ravel()

            #Split to have training and validation data
            x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

            grid = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, scoring='r2', n_jobs=6)
            #n_jobs: take into account number of CPUs to use

            grid.fit(x_train,y_train)

            # print(grid.best_estimator_)
            # print(grid.best_score_)

            #Validation
            y_validation_pred = grid.predict(x_validation)
            r2 = r2_score(y_validation,y_validation_pred)

            if r2 >= threshold:
                x_train = pca_train_mode.to_numpy()

                y_train = dsf_train_mode.iloc[:,pc].to_numpy()
                y_train = y_train.ravel()

                #EOV correction
                y_train_pred = grid.predict(x_train)
                y_pred = grid.predict(x)

                # plt.figure()
                # plt.scatter(y_train,y_train_pred)
                # plt.show()

                dsf_mode.iloc[:,pc] = y - y_pred
                dsf_train_mode.iloc[:,pc] = y_train - y_train_pred

            if file_mode in plot_modes:
                plt.figure()
                plt.scatter(x[:,0],y,c=scenario['DamageScenario'].map(colors),s=s)
                plt.scatter(x_train[:,0],y_train,c='black',s=s)
                plt.show()

            if plot_regression == True:
                col = np.where(r2_mode>=threshold,'green','grey')
                plt.figure(file_mode)
                plt.scatter(np.arange(np.shape(r2_mode)[0]),r2_mode,c=col,s=10)
                plt.plot(np.array([0,np.shape(r2_mode)[0]]),[threshold, threshold],c='black',linestyle='dashed')
                plt.ylim([0,1])
                plt.title(file_mode)
                plt.xlabel('PCs DSF')
                plt.ylabel('R2 Correlation')
                plt.show()

        print('Save files ' + str(file_mode))
        dsf_mode.to_csv('../code_6modes/03regression_analysis/02correction/decision_tree'+'/t'+str(threshold)+'/corrected_dsf_'+file_mode+'.csv',sep=';')
        dsf_train_mode.to_csv('../code_6modes/03regression_analysis/02correction/decision_tree'+'/t'+str(threshold)+'/corrected_dsf_'+file_mode+'_train.csv',sep=';')

    print('done')

#Write command
print('0')
decision_tree(threshold=0.0,plot_modes=[])
decision_tree(threshold=0.05,plot_modes=[])
decision_tree(threshold=0.1,plot_modes=[])
decision_tree(threshold=0.15,plot_modes=[])
decision_tree(threshold=0.2,plot_modes=[])
decision_tree(threshold=0.25,plot_modes=[])
decision_tree(threshold=0.3,plot_modes=[])
decision_tree(threshold=0.35,plot_modes=[])
decision_tree(threshold=0.4,plot_modes=[])
decision_tree(threshold=0.45,plot_modes=[])
decision_tree(threshold=0.5,plot_modes=[])
print('0.5')
decision_tree(threshold=0.55,plot_modes=[])
decision_tree(threshold=0.6,plot_modes=[])
decision_tree(threshold=0.65,plot_modes=[])
decision_tree(threshold=0.7,plot_modes=[])
decision_tree(threshold=0.75,plot_modes=[])
decision_tree(threshold=0.8,plot_modes=[])
decision_tree(threshold=0.85,plot_modes=[])
decision_tree(threshold=0.9,plot_modes=[])
decision_tree(threshold=0.95,plot_modes=[])
decision_tree(threshold=1.0,plot_modes=[])