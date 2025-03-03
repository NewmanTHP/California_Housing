from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor


sns.set(style="whitegrid")


def impute_df_with_knn(df):
    
    # imputation with KNN unsupervised method
    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number])           # select numerical columns in df
    ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist()         # columns w/ nan 
    cols_no_nan = ldf.columns.difference(cols_nan).values     # columns w/o nan 

    for col in cols_nan:                
        imp_test = ldf[ldf[col].isna()]   # indicies which have missing data will become our test set
        imp_train = ldf.dropna()          # all indicies which which have no missing data 
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])
    
    return pd.concat([ldf,ldf_putaside],axis=1)



def correlation_matrix_plot(df):
    ldf = df.select_dtypes(include=[np.number])  
    corr_mat = ldf.corr().round(2)
    _, ax = plt.subplots(figsize=(15,5))
    mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))
    mask = mask[1:,:-1]
    corr = corr_mat.iloc[1:,:-1].copy()
    sns.heatmap(corr,mask=mask,vmin=-1,vmax=1,center=0, 
                cmap='bwr',square=False,lw=2,annot=True,cbar=True)
    ax.set_title('Linear Correlation Matrix')


def pair_plots(df):
    ''' Plots a Seaborn Pairgrid w/ KDE & scatter plot of df features'''
    g = sns.PairGrid(df,diag_sharey=False)
    g.fig.set_size_inches(14,13)
    g.map_diag(sns.kdeplot, lw=2) # draw kde approximation on the diagonal
    g.map_lower(sns.scatterplot,s=15,edgecolor="k",linewidth=1,alpha=0.4) # scattered plot on lower half
    g.map_lower(sns.kdeplot,cmap='plasma',n_levels=10) # kde approximation on lower half
    plt.tight_layout()

    ''' Plot Two Geopandas Plots Side by Side '''


# defining a simple plot function, input list containing features of names found in dataframe
def california_plot(df,lst):

    # load california from module, common for all plots
    cali = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
    cali = cali.assign(area=cali.geometry.area)
    
    # Create a geopandas geometry feature; input dataframe should contain .longtitude, .latitude
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude,df.latitude))
    proj=gplt.crs.AlbersEqualArea(central_latitude=37.16611, central_longitude=-119.44944) # related to view

    ii=-1
    _,ax = plt.subplots(1,2,figsize=(15,5),subplot_kw={'projection': proj})
    for i in lst:

        ii+=1
        tgdf = gdf.sort_values(by=i,ascending=True) 
        gplt.polyplot(cali,projection=proj,ax=ax[ii]) # the module already has california 
        gplt.pointplot(tgdf,ax=ax[ii],hue=i,cmap='magma',legend=True,alpha=1.0,s=3) # 
        ax[ii].set_title(i)

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5)



def gaussian_process_cross_validation(df, target_column, kernel_type='RBF', sample_fraction=0.01, cv=5, seed=303):

    # Optimize data types
    df = df.astype({col: 'float32' for col in df.select_dtypes(include=['float64']).columns})
    
    # Use a smaller subset of the data
    df_sampled = df.sample(frac=sample_fraction, random_state=seed)
    
    X = df_sampled.drop(columns=target_column)  # Explanatory variables
    y = df_sampled[target_column]               # Response variable

    # Define the kernel based on the kernel_type parameter
    if kernel_type == 'RBF':
        kernel = RBF(1.0, (1e-4, 1e1))
    elif kernel_type == 'Matern':
        kernel = Matern(length_scale=1.0, nu=1.5)
    elif kernel_type == 'RationalQuadratic':
        kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    elif kernel_type == 'Exponential':
        kernel = ExpSineSquared(length_scale=1.0, periodicity=3.0)
    else:
        raise ValueError("Unsupported kernel type")

    # Create Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=seed)
    
    # Perform cross-validation
    scores = cross_val_score(gp, X, y, cv=cv, scoring='neg_root_mean_squared_error')

    # Calculate the root mean squared error
    rmse_scores = -scores
    
    # Return the mean and standard deviation of the scores
    print(f"RMSE Scores: {rmse_scores}")
    print(f"Mean: {np.mean(rmse_scores)}")
    print(f"Std: {np.std(rmse_scores)}")