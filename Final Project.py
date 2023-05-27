import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

#Load the dataset
df=pd.read_csv("energydata_complete.csv",header=0)

#Inspecting the data
df.info()
app=df.Appliances
plt.figure(figsize=(16,8))
plt.xticks(np.linspace(0,len(app)+1,2),['2017-12-22','2018-01-11'])
plt.plot(df.date,app)
plt.title("Rented Bike Count vs. Time")
plt.xlabel("Timestamp")
plt.ylabel("Rented number")
plt.tight_layout()
plt.show()

df.cnt.describe()

#Correlation Heatmap
corrmat=df.drop(columns=['timestamp']).corr()
sns.heatmap(corrmat,annot=True)
plt.title("Correlation Matrix of the data")
plt.show()

#ACF of dependent variable
def cal_autocorr(y, n_lag):
    y_mu = np.mean(y)
    r_denominator = np.std(y) ** 2 * (len(y) - 1)
    r_lst = []
    for tau in range(n_lag+1):
        sum = 0
        for t in range(tau,n_lag+1):
            sum += (y[t]-y_mu)*(y[t-tau]-y_mu)
        r_lst.append(sum/r_denominator)
    return r_lst

auto_rent=cal_autocorr(rent,100)
acf_rent=auto_rent[::-1][:-1]+auto_rent
lag=np.arange(-100,101,1)
plt.stem(lag,acf_rent)
m=1.96/np.sqrt(len(auto_rent))
plt.axhspan(-m,m,alpha=0.2, color='blue')
plt.title("autocorrelation funtion of Bike rented Count")
plt.xlabel("Lags")
plt.ylabel("Magnitude")
plt.show()

#Check for colinearity
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
train,test=train_test_split(df,train_size=0.8,random_state=20)
x_train=df.drop(columns=['timestamp','cnt'])
y_train=df.cnt
vif_df=pd.DataFrame()
vif_df['feature'] = df.drop(columns=['timestamp','cnt']).columns
vif_df['VIF'] = [variance_inflation_factor(x_train.values,(i for i in range(len(df.columns))))]
print(vif_df)