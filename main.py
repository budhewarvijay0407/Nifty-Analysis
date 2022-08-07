import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split



data=pd.read_csv('TATAELXSI.csv')
data.set_index('Date',inplace=True,drop=True)
open=data['Open'].tolist()
close=data['Close'].tolist()

list_of_highs=[]
list_of_lows=[]


####Preparing the data for the training#########

sample_size=15
number_of_samples_down=200
number_of_samples_up=80
training_dataset_down=[]
training_dataset_up=[]
for i in range(number_of_samples_down):
    sample=np.random.normal(0.4,0.2,sample_size)
    days_move=[]
    start_value_init = 1
    start_value=1
    for i in sample:
        days_move.append(start_value-start_value_init*i)
        start_value=start_value-0.15
    training_dataset_down.append(days_move)

for i in range(number_of_samples_up):
    sample=np.random.normal(0.4,0.2,sample_size)
    days_move=[]
    start_value_init = 1
    start_value=1
    for i in sample:
        days_move.append(start_value-start_value_init*i)
        start_value=start_value+0.15
    training_dataset_up.append(days_move)

df_down=pd.DataFrame(training_dataset_down)
df_down_transpose=df_down.T
scaler = MinMaxScaler()
y=scaler.fit(df_down_transpose)
y=scaler.transform(df_down_transpose)
df_down=y.T
df_down=pd.DataFrame(df_down)

df_up=pd.DataFrame(training_dataset_up)
df_up_transpose=df_up.T
scaler1 = MinMaxScaler()
y_1=scaler1.fit(df_up_transpose)
y_1=scaler1.transform(df_up_transpose)
df_up=y_1.T
df_up=pd.DataFrame(df_up)
columns_name=[]

for i in range(sample_size):
    columns_name.append('Day'+str(i))

df_down.columns=columns_name
df_up.columns=columns_name

#plt.plot(df_down['Day12'])
#plt.plot(df_up['Day13'])
df_up['output']=1
df_down['output']=0

merged_data=pd.concat([df_up,df_down])
merged_data.reset_index(inplace=True,drop=True)
merged_data=merged_data.sample(frac=1, random_state=1)
#################################Train a MLP algorithm for detection of the upward and downward trend#######################

X=merged_data[merged_data.columns[:len(merged_data.columns)-1]]
y=merged_data[['output']]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,test_size=0.2)
print('Succefully split the train and test dataset')
clf = MLPClassifier(hidden_layer_sizes=(20,),activation = 'logistic',solver='adam',random_state=1).fit(X_train, y_train)
#clf.predict_proba(X_test[:1])
#clf.score(X_test, y_test)