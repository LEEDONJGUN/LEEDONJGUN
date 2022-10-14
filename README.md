"""
 * Coyright (c) 2022 CHANGHUN LEE, CAE Team Samsung Display Rearch Center. All rights reserved.
 * NOTICE:  This program and the accompanying codes are available under the agreement of CAE.
 * Contributors: CHANGHUN LEE - Intial Coding and implementation

@author: ch78.lee
"""
# In[]:
import os
import time 
#import numpy as np
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt    
import seaborn as sns
import joblib
#import xgboost as xgb

from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
HOME_DIR = os.getcwd()
os.chdir(HOME_DIR)
plt.rcParams['axes.unicode_minus'] = False
sns.set(font="Malgun Gothic")

# In[]:
def one_hot (df,ohe_list):
    enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
    onehot_col = df[ohe_list] #One hot encoding할 컬럼명
    enc.fit(onehot_col)
    oh = enc.fit_transform(onehot_col)
    column_list = enc.get_feature_names(ohe_list)
    for i in range(len(column_list)): #컬명명의 One hot encoding 진행
        onehot_col[column_list[i]]=oh[:,i]
    onehot_col = onehot_col.drop(ohe_list,axis=1)
    raw_data=pd.concat([df,onehot_col],axis=1)
    ohe_col_list = onehot_col.columns.tolist()
    joblib.dump(enc, 'one_hot_encoding.pkl')
    return raw_data, ohe_col_list #dataframe과 onehot 컬럼 반환

def one_hot_prediction (df,col_list):
    onehot_apply = df[ohe_list]
    oh_real = ohe_model.transform(onehot_apply)
    column_list = ohe_model.get_feature_names(ohe_list)
    for i in range(len(column_list)):
        onehot_apply[column_list[i]]=oh_real[:,i]
    onehot_apply = onehot_apply.drop(ohe_list, axis=1)
    numeric_data = df[col_list]
    X = pd.concat([numeric_data,onehot_apply],axis=1)
    real_X_scaled = scaler_model.transform(X)
    for i in range(no_model):
        model = joblib.load(str(i+1)+"_predict_model.pkl")
        predict_Y = model.predict(real_X_scaled)
        df[str(i+1)+"_Dose_Pred"] = predict_Y
    return df, real_X_scaled, onehot_apply

def data_rename (data):
    data.rename(columns = 
        {
         'EQP ID':'EQP_ID',
         'LOT ID':'LOT_ID',
         'GLASSID_CHKSUM':'GLASS_ID',
         'GLASS ID CHKSUM':'GLASS_ID',
         '측정 CD':'Last_CD',
         '노광량':'DOSE_AVG',
         'PHOTO 진행시간':'Date',
         'PHT 진행 시각':'Date',
         'MASK':'MASK_ID',
         'MASK ID':'MASK_ID',
         'MASKTYPE':'MASK_type',
         'MASK 종류':'MASK_type',
         'APC_TARGET':'APC_target',
         'ADI Target\r\n(APC)':'APC_target',
         '설계치 ADI\r\n(PDR)':'PDR',
         'Global bias':'Bias',
         'PR_재료':'PR',
         'PR종류':'PR',   
         'TPR(KÅ)':'TPR',
         'DEV_Target':'Time',
         'DEV Time (s)':'Time',
         '-MTT':'MTT',
         'MASK 계산값':'MASK_Cal',
         'MASK-ADI(APC)':'MASK_ADI'
         }, 
        inplace=True)
    return data

def data_add_col (df):
    df['rate'] = df['TPR']/df['Time']
    df['Layer'] = df.PPID.str.split('_').str[3]
    df['MASK_model'] = df.MASK_ID.str.split('-').str[0]
    return df
    
def col_strip (df):
    for i in range(len(strip_col)):
        df[strip_col[i]] = df[strip_col[i]].str.strip()
    return df

def change_date (df):
    df["Date"] = [w.replace('오전', 'AM') for w in df["Date"]]
    df["Date"] = [w.replace('오후', 'PM') for w in df["Date"]]
    df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d %p %I:%M:%S")
    return df  


# In[]:
file_name1 = 'raw_data' #학습 data set 파일 이름 입력
#raw_data = pd.read_table(r"{}\{}.txt".format(HOME_DIR, file_name1), delimiter='\t', encoding='utf-16-le', engine='python')
raw_data = pd.read_table(r"{}\{}.txt".format(HOME_DIR, file_name1), delimiter='\t', encoding='utf-8-sig', engine='python')
print("Training data shape is..." , raw_data.shape)
print('\n')

file_name2 = '9e' #예측 data set 파일 이름 입력 , #기존 data format 유지
sample= pd.read_table(r"{}\{}.txt".format(HOME_DIR, file_name2), delimiter='\t', encoding='utf-8-sig', engine='python')
print("Prediction data shape : ", sample.shape)
print('\n')

# In[]:
no_model = 20 #생성 모델 수 
offset_glass_count = 5 # offset 계산시 필요 Glass
ohe_col_check = 1 #1이면 학습 data기준 column check, 0이면 무조건 예측

feature_col = ['EQP_ID','PPID','Date','MASK_ID','DOSE_AVG','LOT_ID','GLASS_ID','Last_CD','MASK_type','APC_target','TPR','PR','Time','MTT']
target = 'DOSE_AVG'

#ohe_list = ['EQP_ID','MASK_type','PR','Layer','MASK_model']
ohe_list = ['EQP_ID','MASK_type','PR','Layer']
ohe_col_count = len(ohe_list)
numeric_col_list = ['rate','MTT','Last_CD']
sample_col_list = ['rate','MTT','APC_target']
strip_col = ['EQP_ID','PPID','MASK_type','PR']

#numeric_col_list = ['PDR','Bias','TPR','Time','rate','MTT','MASK_Cal','MASK_ADI','Last_CD']
#sample_col_list = ['PDR','Bias','TPR','Time','rate','MTT','MASK_Cal','MASK_ADI','APC_target']

sample = data_rename(sample)
sample = col_strip (sample)
sample = data_add_col(sample)

layer_col = sample['Layer'].unique()

#model = xgb.XGBRegressor()
model = RandomForestRegressor()
kfold = KFold(n_splits=no_model)

print("Number of ensemble model  = ",no_model)
print("EQP Offset Glass Count = ",offset_glass_count)
print('\n')

# In[]: Data 전처리 및 Filter
data = data_rename(raw_data)
data = col_strip (data)

data = data[data['DOSE_AVG'] > 10] # 최소 Dose값 설정으로 이상치 제거
data = data[data['MTT'].between(-5, 5)] #MTT 이상치 제거

data = data[feature_col]
data = data_add_col(data)
data = change_date(data)

data = data.sample(frac=1, random_state=0)
#data = data.dropna() #null value 제거
data = data.dropna(subset=['rate', 'MTT', 'Last_CD','APC_target','PR']) #null value 제거, 비어 있을 가능성 있는 컬럼

data = data[data['Layer'].isin(layer_col)]

print("data shape...  = ", data.shape)
print('\n')

data_columns = data.columns.tolist()
data = data.drop_duplicates(data_columns, keep='first') #중복된 data 제거

print('Layer counts')
print(data.groupby('Layer')['DOSE_AVG'].count())
print('\n')

data = data.reset_index(drop=True, inplace=False)

print("Preprocessing training data shape...  = ", data.shape)
print('\n')

# In[]:
if 0: # pair plot 그릴시 "1"로 설정 "0"이면 안 그림
    pair_col = numeric_col_list + [target]
    pairplot_data = data[pair_col]
    
    plt.figure(figsize=(4,4))
    sns.pairplot(pairplot_data)
    plt.show()

# In[]:
offset_data = data.sort_values(by="Date", ascending=False).groupby("EQP_ID").head(offset_glass_count)
offset_data = offset_data.sort_values(by="EQP_ID", ascending=True)
model_data = data
#model_data = data.drop(offset_data.index)
model_data = model_data.reset_index(drop=True, inplace=False)
offset_data = offset_data.reset_index(drop=True, inplace=False)


# In[]:
ohe_data, ohe_col_list = one_hot(model_data,ohe_list)
ohe_input_col_list = numeric_col_list + ohe_col_list

input_data = ohe_data[ohe_input_col_list]
output_data = data[target]

scaler = StandardScaler()
scaler.fit(input_data)
joblib.dump(scaler, "scaler.pkl")
X_scaled = scaler.transform(input_data)
y = output_data

# In[]:
One_cls_svm = OneClassSVM(gamma='auto', nu=0.2)
One_cls_svm.fit(X_scaled)
joblib.dump(One_cls_svm, "One_cls_svm.pkl")

# In[]:
n=1
print("Dose prediction model 생성 Start!!!")
for train_index, test_index in kfold.split(input_data):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    x_train, y_train = X_scaled[train_index], y[train_index]
    model.fit(x_train,y_train)
    pred_train = model.predict(x_train)
    r2_train = r2_score(y_train,pred_train)
    joblib.dump(model, str(n)+"_predict_model.pkl")
    print(str(n),"/",str(no_model)," Model R2 is...", str(round(r2_train,ndigits=3)))
    n=n+1
print("Dose prediction model 생성 완료")
print('\n')

# In[]:
ohe_model = joblib.load("one_hot_encoding.pkl")
scaler_model = joblib.load("scaler.pkl")
One_cls_svm_model = joblib.load("One_cls_svm.pkl")

# In[]:
offset_data, offset_X_scaled, offset_ohe = one_hot_prediction(offset_data,numeric_col_list)
offset_data['MED_Dose_Pred'] = offset_data.iloc[:,-no_model:].median(axis=1)
offset_data['offset_k'] = offset_data['DOSE_AVG']-offset_data['MED_Dose_Pred']

# In[]:
grouped = offset_data['offset_k'].groupby(offset_data['EQP_ID'])
eqp_offset = grouped.mean()
eqp_offset = eqp_offset.to_frame(name='offset_k')
eqp_offset = eqp_offset.reset_index()

# In[]:
sample, sample_X_scaled, sample_ohe = one_hot_prediction(sample,sample_col_list)
sample_ohe['sum_count'] = sample_ohe.iloc[:,:].sum(axis=1)

max_min = sample.iloc[:,-no_model:].max(axis=1)-sample.iloc[:,-no_model:].min(axis=1)
std_dose_pred = sample.iloc[:,-no_model:].std(axis=1)
med_dose_pred = sample.iloc[:,-no_model:].median(axis=1)

sample['Max-Min_Dose_Pred'] = max_min
sample['STD_Dose_Pred'] = std_dose_pred
sample['MED_Dose_Pred'] = med_dose_pred
sample.insert(sample.shape[1],'MED_offset',sample['EQP_ID'].map(eqp_offset.set_index('EQP_ID')['offset_k']))

for i in range(0,len(sample)):
    mask_bool = data['MASK_model'].isin([sample.loc[i,'MASK_model']]).value_counts()
    sample.loc[i,'MASK_model_check'] = np.where(mask_bool[0] == data.shape[0], 'Unused', 'Used')

Osvm_result = One_cls_svm_model.predict(sample_X_scaled)
sample["OSVM_result"] = Osvm_result

sample['MED_offset'] = sample['MED_offset'].fillna(value=0)
sample['Dose_final'] = sample['MED_Dose_Pred'] + sample['MED_offset']

if ohe_col_check == 1:    
    for i in range(0,len(sample)):
        sample.loc[i,'Dose_final'] = np.where(sample_ohe.loc[i,'sum_count'] == ohe_col_count, round(sample.Dose_final[i],ndigits=2), 'NaN')
    pass 

# In[ ]:
sel_col = ['EQP_ID','PPID','Layer','STD_Dose_Pred','OSVM_result','Dose_final']
result_table = sample[sel_col]
print("========================== Dose Prediction Table ==========================")
print(result_table)

sample_filename = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '_Dose Prediction result data.csv'
sample.to_csv(sample_filename,index=True,encoding='utf-8-sig')


