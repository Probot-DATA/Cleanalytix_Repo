import pandas as pd
import numpy as np
datasets={
           "adult":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\adult\adult.data",
           "adult_test":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\adult\adult.test",
           "customer_churn":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv",
           "UCI_credit_card":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\UCI_credit_card\UCI_Credit_Card.csv",
           "yellow1":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\yellow_trip_data\yellow_tripdata_2015-01.csv",
           "yellow2":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\yellow_trip_data\yellow_tripdata_2016-01.csv",
           "yellow3":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\yellow_trip_data\yellow_tripdata_2016-02.csv",
           "yellow4":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\yellow_trip_data\yellow_tripdata_2016-03.csv",
           "CC_General":r"C:\Users\Prabhat\Desktop\Carear\projects\project-1\data\CC GENERAL.csv"
           }
dataset_names=['yellow' , 'UCI_credit_card' , 'customer_churn' , 'adult','CC_General']

dataset_names
chunk_size=20000
customer_churn=pd.read_csv(datasets["customer_churn"])
UCI_credit_card=pd.read_csv(datasets["UCI_credit_card"])
CC_General=pd.read_csv(datasets["CC_General"])  
chunk1=pd.read_csv(datasets["yellow1"],chunksize=chunk_size)
yellow1 = next(chunk1) 
chunk2=pd.read_csv(datasets["yellow2"],chunksize=chunk_size)
yellow2 = next(chunk2) 
chunk3=pd.read_csv(datasets["yellow3"],chunksize=chunk_size)
yellow3 = next(chunk3) 
chunk4=pd.read_csv(datasets["yellow4"],chunksize=chunk_size)
yellow4 = next(chunk4)
yellow= pd.concat([yellow1, yellow2, yellow3, yellow4], ignore_index=True)
yellow

yellow
yellow = yellow.rename(columns={
    'RateCodeID': 'RateCodeID_1',
    'RatecodeID': 'RateCodeID_2'
})
yellow['RateCodeID_Final'] = yellow['RateCodeID_1'].combine_first(yellow['RateCodeID_2'])
yellow = yellow.drop(['RateCodeID_1', 'RateCodeID_2'], axis=1)

col_names = ['age','workclass','fnlwgt','education','education-num',
             'marital-status','occupation','relationship','race','sex',
             'capital-gain','capital-loss','hours-per-week','native-country','<=50k']

adult = pd.read_csv(datasets["adult"], header=None, names=col_names)
adult_test = pd.read_csv(datasets["adult_test"], header=None, names=col_names, skiprows=1)

adult = pd.concat([adult, adult_test], ignore_index=True)

yellow['tpep_dropoff_datetime']=pd.to_datetime(yellow['tpep_dropoff_datetime'])
yellow['tpep_pickup_datetime']=pd.to_datetime(yellow['tpep_pickup_datetime'])


dataset_list=[yellow , UCI_credit_card , customer_churn,adult,CC_General]

        
    

from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas.api.types import is_numeric_dtype
def get_metrics(df_list,dataset_name_list):
    data=[]
    for df,dataset_name in zip(df_list,dataset_name_list):
        df=pd.DataFrame(df)
        #duplicate row percentage
        duplicate_row_percentage=((len(df)-len(df.drop_duplicates()))/len(df))*100
        for column in df.columns:
            #Meta info-:
            dataset_name=dataset_name
            column_name=column
            dtype=df[column].dtype
        
            #completness-:
            non_null_count=df[column].notnull().sum()
            missing_count=df[column].isnull().sum()
            missing_percent=(missing_count/len(df[column]))*100
        
            #Uniquness and variety
        
            unique_count=df[column].nunique()
            unique_percent=(unique_count/len(df[column]))*100
            mode=df[column].mode()[0]
            mode_percent=((df[column]==mode).sum()/len(df[column]))*100
        
        
            #based on column types
        
            if is_numeric_dtype(df[column]):
                mean=df[column].mean()
                std=df[column].std()
                min=df[column].min()
                max=df[column].max()
                median=df[column].median()
                skewness=df[column].skew()
                abs_skew = abs(skewness)
                if abs_skew < 0.5:
                    skewness_level = "low"
                elif abs_skew < 1:
                    skewness_level = "moderate"
                else:
                    skewness_level = "high"
                #outliers using z-score
                scalar=StandardScaler()
                z_score=scalar.fit_transform(df[[column]])
                outliers=int((np.abs(z_score)>=3).sum())
            
                num_categories=None
                rare_category_percent=None
                whitespace_issues=None
                min_date=None
                max_date=None
                date_range_days=None
            elif((df[column].dtype=='object') or (df[column].dtype=='bool') or (df[column].dtype=='category') ):
                num_categories=unique_count
                unique=list(df[column].unique())
                rare_category_percent=100
                for item in unique:
                    item_count=(df[column]==item).sum()
                    item_percentage=(item_count/len(df[column]))*100
                    if item_percentage<=5.5 and (item_percentage<rare_category_percent):
                        rare_category_percent=item_percentage
                if(rare_category_percent==100):
                    rare_category_percent=None
                temp1=df[column].str.startswith(" ").sum()
                temp2=df[column].str.endswith(" ").sum()
                whitespace_issues=temp1+temp2
                mean=None
                std=None
                min=None
                max=None
                median=None
                outliers=None
                min_date=None
                max_date=None
                date_range_days=None
                skewness=None
                skewness_level=None
            else:
                min_date=df[column].min()
                max_date=df[column].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range_days = (max_date - min_date).days
                else:
                    date_range_days = None
                mean=None
                std=None
                min=None
                max=None
                median=None
                outliers=None
                num_categories=None
                rare_category_percent=None
                whitespace_issues=None
                skewness=None
                skewness_level=None
            data.append([dataset_name,column_name,dtype,non_null_count,missing_count,missing_percent,unique_count,unique_percent,mode,mode_percent,mean,std,skewness,skewness_level,min,max,median,outliers,num_categories,rare_category_percent,whitespace_issues,min_date,max_date,date_range_days,duplicate_row_percentage])
    return data        
                    
                
            
            
            
            
            

        
        
        
        
from collections import Counter
import math
def get_probs(df, column , bins):
    probabilities=[]
    if is_numeric_dtype(df[column]):
        bin_indices = pd.cut(
        df[column],
        bins=bins,
        include_lowest=True,
        labels=False
        ) + 1
        counter=Counter(bin_indices)
        for i in range(1,len(bins),1):
            count=counter.get(i,0)
            prob=(count)/(len(df[column])) #adding 1 for zero error handling
            prob=max(prob,0.0000001)
            probabilities.append(prob)
    else:
        counter=Counter(df[column])
        for cat in bins:
            count=counter.get(cat, 0)
            prob=count / len(df[column])
            prob=max(prob, 0.0000001)
            probabilities.append(prob)
    return probabilities

def KL_PSI_and_JS(df_base, df_new , column ):
        if is_numeric_dtype(df_base[column]):
            if len(df_base[column])<10000:
                bins = np.quantile(df_base[column], q=np.linspace(0,1,6))
                bins = np.unique(bins)
                bins = np.sort(bins)
                if len(bins) < 2:
                    return 0, 0, 0
            elif len(df_base[column])<50000:
                bins = np.quantile(df_base[column], q=np.linspace(0,1,11))
                bins = np.unique(bins)
                bins = np.sort(bins)
                if len(bins) < 2:
                    return 0, 0, 0
            else:
                bins = np.quantile(df_base[column], q=np.linspace(0,1,21))
                bins = np.unique(bins)
                bins = np.sort(bins)
                if len(bins) < 2:
                    return 0, 0, 0
        else:
            bins=df_base[column].unique()
        probs_base=get_probs(df_base , column, bins)
        probs_new=get_probs(df_new , column , bins)
        KL_div=0
        PSI_div=0
        JS_div=0
        probs_base=np.array(probs_base)
        probs_new=np.array(probs_new)
        M=(probs_new+probs_base)/2
        for i in range(0,len(probs_base) , 1):
            val_KL=probs_base[i]*(math.log(probs_base[i]/probs_new[i]))
            val_PSI=(probs_base[i]-probs_new[i])*(math.log(probs_base[i]/probs_new[i]))
            val_JS=(probs_base[i]*(math.log(probs_base[i]/M[i]))+probs_new[i]*(math.log(probs_new[i]/M[i])))/2
            KL_div+=val_KL
            PSI_div+=val_PSI
            JS_div+=val_JS
        return KL_div , PSI_div,JS_div
    

def get_entropy(df_base,column):
    if is_numeric_dtype(df_base[column]):
        if len(df_base[column])<10000:
            bins = np.quantile(df_base[column], q=np.linspace(0,1,6))
            bins = np.unique(bins)
            bins = np.sort(bins)
            if len(bins) < 2:
                return 0
        elif len(df_base[column])<50000:
            bins = np.quantile(df_base[column], q=np.linspace(0,1,11))
            bins = np.unique(bins)
            bins = np.sort(bins)
            if len(bins) < 2:
                return 0
        else:
            bins = np.quantile(df_base[column], q=np.linspace(0,1,21))
            bins = np.unique(bins)
            bins = np.sort(bins)
            if len(bins) < 2:
                return 0
    else:
        bins=df_base[column].unique()
    probs=get_probs(df_base,column,bins)
    entropy=0
    for prob in probs:
        entropy-=(prob*(math.log(prob)))
    return entropy

def get_type_error_count(df, column):
    expected_type=df[column].dtype
    error_count = 0
    for val in df[column]:
        if is_numeric_dtype(df[column]):
            try:
                float(val)  
            except (ValueError, TypeError):
                error_count += 1
        elif expected_type == "datetime64[ns]":
            try:
                pd.to_datetime(val)
            except (ValueError, TypeError):
                error_count += 1
        elif ((expected_type == "category") or (expected_type=='object')):
            if not isinstance(val, str):
                error_count += 1
    return error_count

def rule_based_errors(df,column,rules={}):
    errors=0
    rules=dict(rules)
    if column in rules.keys():
        for val in df[column]:
            if not(rules[column](val)):
                errors+=1
        return errors
    else:
        return 0
def check_age(val):
    if val>0:
        return True
    else:
        return False
rules={
    "age": check_age
    }   #column_name : function_name

       
def advanced_metrics(dataset_name_list,df_base_list,df_new_list):
    metrics=[]
    for df_base,df_new,dataset_name in zip(df_base_list,df_new_list,dataset_name_list):
        df_base=pd.DataFrame(df_base)
        df_new=pd.DataFrame(df_new)
        for column in df_base.columns:
            KL_div , PSI_div,JS_div =KL_PSI_and_JS(df_base , df_new,column)
            entropy=get_entropy(df_base,column)
            type_errors=get_type_error_count(df_base,column)
            rule_errors=rule_based_errors(df_base,column,rules)
        
            #more adv metrics soon
            metrics.append([dataset_name,column,KL_div,PSI_div,JS_div,entropy,type_errors,rule_errors])
    return metrics
            
            
            
            
            
            
        
        
    
from sklearn.model_selection import train_test_split
def generate_meta(dataset_name_list,dataset_list,new_dataset_list=None):
    data=get_metrics(dataset_list,dataset_name_list)
    df_old_list=[]
    df_new_list=[]
    if new_dataset_list is None:
        for df in dataset_list:
            df_old , df_new =train_test_split(df,test_size=0.3,random_state=1)
            df_old_list.append(df_old)
            df_new_list.append(df_new)
    else:
        if len(new_dataset_list) != len(dataset_list):
            raise ValueError("new_dataset_list must match dataset_list length")
        for i, (ref_df, new_df) in enumerate(zip(df_old_list, df_new_list)):

            if list(ref_df.columns) != list(new_df.columns):
                raise ValueError(
                f"Column mismatch in dataset index {i}"
            )

            if not all(ref_df.dtypes == new_df.dtypes):
                raise ValueError(
                f"Dtype mismatch in dataset index {i}"
            )
        df_old_list = [df for df in dataset_list]
        df_new_list = [df for df in new_dataset_list]
    metrics=advanced_metrics(dataset_name_list,df_old_list,df_new_list)
    meta_data=pd.DataFrame(data=data,columns=[
    # ù Meta Info ù
    "dataset_name",        
    "column_name",         
    "dtype",               

    # ù Completeness ù
    "non_null_count",      
    "missing_count",       
    "missing_percent",     

    # ù Uniqueness & Variety ù
    "unique_count",        
    "unique_percent",      
    "mode",                
    "mode_percent",        

    # ù Numeric Columns ù
    "mean",                
    "std",
    "skewness",
    "skewness_level",                 
    "min",                 
    "max",                 
    "median",              
    "outlier_count",       

    # ù Categorical Columns ù
    "num_categories",      
    "rare_category_percent",
    "whitespace_issues",   

    # ù Datetime Columns ù
    "min_date",            
    "max_date",            
    "date_range_days",     

    # ù Quality & Resource ù
    "duplicate_rows_percent"    
])
    meta_data.replace({pd.NaT: np.nan}, inplace=True)
    meta_table_adv=pd.DataFrame(data=metrics,columns=['dataset_name','column_name','KL_divergence','PSI_divergence','JS_divergence','Entropy','Type_Errors','rule_errors'])
    meta_table_combined=pd.merge(meta_data,meta_table_adv,on=['dataset_name', 'column_name'],how='inner')
    return meta_table_combined
# meta_table_combined=generate_meta(dataset_names,dataset_list) #step 1    
    
    
    
    
    


def summarize_dataset_health(meta_table):
    summary = meta_table.groupby("dataset_name", as_index=False).agg({
        "missing_percent": "mean",
        "duplicate_rows_percent": "mean",
        "outlier_count": np.nansum,
        "whitespace_issues": np.nansum
    })

    summary.rename(columns={
        "missing_percent": "avg_missing_percent",
        "duplicate_rows_percent": "avg_duplicate_percent",
        "outlier_count": "total_outliers",
        "whitespace_issues": "total_whitespace_issues"
    }, inplace=True)

    return summary


# meta_table_combined.to_csv('metrics.csv',index=False)
import pandas as pd

def cleaning_recommendations(meta_data_combined):
    recs = []
    Thresholds = {
    "drop_missing": 20,
    "rare_category": 5,
    "small_date_range": 1
}
    for i, row in meta_data_combined.iterrows():
        suggestions = []
        if row['missing_percent'] > Thresholds['drop_missing']:
            suggestions.append("Consider dropping column due to high missing values")
            
        else:
            if pd.notna(row['whitespace_issues']) and row['whitespace_issues'] > 0:
                suggestions.append("Strip leading/trailing spaces")
            if pd.notna(row['Type_Errors']) and row['Type_Errors'] > 0:
                suggestions.append("Type mismatch detected")
            if pd.notna(row['rule_errors']) and row['rule_errors'] > 0:
                suggestions.append("Business rule violated (default is null)")
            if pd.notna(row['rare_category_percent']) and row['rare_category_percent'] > Thresholds['rare_category']:
                suggestions.append("Combine rare categories or resolve typos")
            if row['dtype'] == 'datetime64[ns]' and pd.notna(row['date_range_days']) and row['date_range_days'] < 1:
                suggestions.append("Check for incorrect or same date entries")
            if pd.notna(row['outlier_count']) and row['outlier_count'] > 0:
                suggestions.append("Handle outliers (default is mean)")
            if 0 < row['missing_percent'] <= Thresholds['drop_missing']:
                suggestions.append("Impute missing values (default is mean for numeric columns and mode for categorical columns)")
            if row['skewness_level'] == 'high':
                suggestions.append("Highly skewed distribution; consider power transformation for distance-based models")
        if len(suggestions) == 0:
            suggestions.append("No major issues detected")
            
        recs.append({
            "dataset_name": row["dataset_name"],
            "column_name": row["column_name"],
            "recommendations": ", ".join(suggestions)
        })

    return pd.DataFrame(recs)
# recommendations = cleaning_recommendations(meta_table_combined)

from nltk import ngrams
from nltk.metrics.distance import jaccard_distance
from sklearn.preprocessing import PowerTransformer


def clean(df, column, clean_type,replacement=None):
    def clean_outliers(df, column):
        while True:
            col = df[column]
            if col.std() == 0:
                break
            z = ((col - col.mean()) / col.std()).abs()
            outlier_idx = z >= 3
            if outlier_idx.sum() == 0:
                break
            if replacement:
                df.loc[outlier_idx, column] = replacement
            else:
                df.loc[outlier_idx, column] = col.median()
        return df[column]

    def keep_string_else_none(val):
        if isinstance(val, str):
            return val
        else:
            return None
        #1 Drop column due to high missing values
    if clean_type == 1:
        df.drop(columns=[column], inplace=True)
        return None
        #2 Impute missing values
    elif clean_type == 2:
        if replacement:
            df[column] = df[column].fillna(replacement)
        else:
            if (is_numeric_dtype(df[column]) and (df[column].dtype != 'bool')) :
                df[column] = df[column].fillna(df[column].mean())
            elif df[column].dtype=='datetime64[ns]':
                df[column]=df[column].fillna(df[column].mode()[0])
            else:
                df[column]=df[column].fillna(df[column].mode()[0])

        #3 Handle outliers using Z-score method
    elif clean_type == 3:
        df[column] = clean_outliers(df, column)  
        #4 Strip leading/trailing spaces
    elif clean_type == 4:
        def striping(value):
            if pd.isna(value):
                return value
            return str(value).strip()
        df[column] = df[column].apply(striping)

        #5 Combine rare categories and resolve typos
    elif clean_type == 5:
        condition=False
        breakpoint=0
        substitutes={}
        if df[column].dtype == 'object' or df[column].dtype == 'category':
            series=df[column].value_counts()
            for i,(k,v) in enumerate(series.items()):
                v=(v/len(df[column]))*100
                if v<=5:#checking if its a typo
                    if not (condition):
                        breakpoint=i
                        condition=True
                    if len(str(k))>=3:
                        k_ngrams=ngrams(str(k).lower(),3)
                        k_set=set()
                        for gram in k_ngrams:
                            k_set.add(" ".join(gram))
                        
                        for index,key in enumerate(series.index):
                            if index==breakpoint:
                                break
                            key_ngrams=ngrams(str(key).lower(),3)
                            key_set=set()
                        
                            for gram in key_ngrams:
                               key_set.add(" ".join(gram))
                            if len(k_set.union(key_set)) == 0:
                                distance = 1
                            else:
                                distance = jaccard_distance(k_set, key_set)
                            if distance<0.25:
                                substitutes[k]=key
                            else:
                                substitutes[k]='Other'#rare category 
            df[column]=df[column].replace(substitutes)               
        #6 Check for incorrect/same date entries
    elif clean_type == 6:
        if ((df[column].dtype=='datetime64[ns]') and (df[column].nunique()<=1)):
            print( column , "has identical or invalid date entries not required for data analysis")
    #7 resolve missmatches and convert to nan
    elif clean_type == 7:
        expected_type = df[column].dtype
        if is_numeric_dtype(expected_type):
            df[column]=pd.to_numeric(df[column], errors='coerce')
        elif expected_type=='datetime64[ns]':
            df[column]=pd.to_datetime(df[column], errors='coerce')
        elif expected_type in ['object', 'category']:
            df[column]=df[column].apply(keep_string_else_none)
            
    #8 replace invalid buisness values (avoiding hallucinating of values)
    elif clean_type==8:
        func=rules[column]
        for i,val in enumerate(df[column]):
            if replacement:
                if not(func(val)):
                    df.loc[i,column]=replacement
            else:
                if not(func(val)):
                    df.loc[i,column]=None
    #9 Highly skewed distribution; consider power transformation for distance-based models
    elif clean_type == 9:
        pt = PowerTransformer(method="yeo-johnson")
        df[column] = pt.fit_transform(df[[column]])
        
    elif clean_type == 10:
        pass
    return df[column]

import copy
def get_cleaned_data(dataset_name_list,dataset_list,recommendations):
    mapped_suggestions={
        1:"Consider dropping column due to high missing values" ,
        2:"Impute missing values (default is mean for numeric columns and mode for categorical columns)",
        3:"Handle outliers (default is mean)",
        4:"Strip leading/trailing spaces",
        5:"Combine rare categories or resolve typos",
        6:"Check for incorrect or same date entries",
        7:"Type mismatch detected",
        8:"Business rule violated (default is null)",
        9:"Highly skewed distribution; consider power transformation for distance-based models",
        10:"No major issues detected"
    }
    NEEDS_REPLACEMENT = {2, 3, 8}  # 2 = impute, 3 = outliers, 8 = business rule
    dataset_list_cleaned=[]
    change_log=[]
    for item,dataset_name in zip(dataset_list,dataset_name_list):
        item=item.copy(deep=True)
        for column in item.columns:
            suggesions=recommendations[recommendations['column_name']==column]['recommendations'].iloc[0]
            suggesions=str(suggesions).split(',')
            for suggesion in suggesions:
                suggesion = str(suggesion).strip()

                clean_type = None
                for k, v in mapped_suggestions.items():
                    if suggesion == v.strip():
                        clean_type = k
                        break
                replacement = None
                user_val="default"
                if clean_type in NEEDS_REPLACEMENT:
                    col_dtype = item[column].dtype
                    # print(f"\nColumn: {column}")
                    # print(f"Suggested fix: {suggesion}")
                    # print(f"Column type: {col_dtype}")
                    # user_val = input("Enter replacement value (or type 'default'): ").strip()
                    user_val="default"
                if user_val.lower() != "default":
                    try:
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            replacement = float(user_val)
                        else:
                            replacement = user_val
                    except:
                        print("Invalid value. Using default.")
                        replacement = None
                else:
                    replacement = None   # tells clean() to use default

                before = item[column].copy()
                result = clean(item, column, clean_type, replacement)

                if result is not None:
                    item[column] = result

                affected = (before != result).sum()

                change_log.append({
        "dataset": dataset_name,
        "column": column,
        "rule_id": clean_type,
        "rule_name": mapped_suggestions[clean_type],
        "replacement": replacement if replacement is not None else "default",
        "affected_rows": int(affected)
    })

        item=item.drop_duplicates()
        dataset_list_cleaned.append(item)
    return dataset_list_cleaned , change_log 
# dataset_list_cleaned=get_cleaned_data(dataset_list,recommendations) #step 3             
            
    
# meta_table_combined_cleaned=generate_meta(dataset_names,dataset_list_cleaned)# step 4 optional  


def get_table_for_DQ_computation(meta_table):
    summarized_meta=meta_table.drop(columns=['dtype','non_null_count','missing_count','unique_count','mode','mode_percent','mean','std','min','max','median','num_categories','min_date','max_date','date_range_days','Entropy'])
    table_lengths={}
    for index,data in enumerate(dataset_list):
        table_lengths[dataset_names[index]]=len(data)
    outlier_percent=[]
    whitespace_percent=[]
    rule_errors_percent=[]
    for index,val in summarized_meta.iterrows():
        outlier_percent.append((val['outlier_count']/table_lengths[val['dataset_name']])*100)
        whitespace_percent.append((val['whitespace_issues']/table_lengths[val['dataset_name']])*100)
        rule_errors_percent.append((val['rule_errors']/table_lengths[val['dataset_name']])*100)
    summarized_meta['outlier_percent']=outlier_percent
    summarized_meta['whitespace_percent']=whitespace_percent
    summarized_meta['rule_errors_percent']=rule_errors_percent
    summarized_meta.drop(columns=['outlier_count','whitespace_issues','rule_errors','Type_Errors','unique_percent','rare_category_percent','KL_divergence'],inplace=True)
    return summarized_meta
# summarized_meta_table=get_table_for_DQ_computation(meta_table_combined)#step 5
# summarized_meta_table_cleaned=get_table_for_DQ_computation(meta_table_combined_cleaned)#step 6




    

def get_weights(dataset_name,summarized_meta):
    metrics=list(summarized_meta.columns)
    metrics.remove('skewness')
    metrics.remove('skewness_level')
    metrics.remove('dataset_name')
    metrics.remove('column_name')
    #while True:
        #print(f'enter your weights for each metric for {dataset_name} , they must sum to 1 , else metrics wont be considered\n')
    weights=[1/len(metrics)]*len(metrics)
        # for i,metric in enumerate(metrics):
        #     print("enter the weight for the following metric\n",metric+'\n','default value is ',1/len(metrics),'\n','press enter for default\n')
        #     weight=input()
        #     weight=weight.strip()
        #     if weight == "":
        #         continue
        #     elif weight.replace('.', '', 1).isdigit():
        #         weights[i] = float(weight)
        #     else:
        #         print("Invalid input, keeping default.\n")
        # weight_sum=sum(weights)
        # if abs(weight_sum - 1) < 1e-6:
        #     print("weights accepted")
        #     return weights
        #     break
        # else:
        #     print('weights not accepted , sum not 100 , please try again\n')
    return weights
    

    


    
from scipy.stats import percentileofscore

def get_normalized_divergence(temp,metric):
    if metric=='PSI_divergence':
        vals=[]
        for index,row in temp.iterrows():
            value=row[metric]
            if(value<0.1):
                vals.append(1)
            elif(value>0.1 and value<0.25):
                vals.append(0.4)
            else:
                vals.append(0)
        temp[metric]=vals
        return temp[metric]
    else:
        js_penalties = []
        for dataset, group in temp.groupby("dataset_name",sort=False):
            js_vals = group[metric].values
            js_percentiles = [
                percentileofscore(js_vals, v, kind="weak") / 100
                for v in js_vals
                ]
            for js_val, js_pct in zip(js_vals, js_percentiles):
                if js_val < 0.01:          # negligible JS
                    js_penalties.append(1)
                else:
                    js_penalties.append(1-js_pct)
        temp[metric]=js_penalties
    return temp[metric]
                        
def get_new_values(summarized_meta_table):
    metrics=list(summarized_meta_table.columns)
    metrics.remove('dataset_name')
    metrics.remove('column_name')
    metrics.remove('skewness')
    metrics.remove('skewness_level')
    temp=summarized_meta_table.copy(deep=True)
    for i,metric in enumerate(metrics):
        if (metric=='PSI_divergence') or (metric=='JS_divergence'):
            temp[metric]=get_normalized_divergence(temp,metric)
        else:
            vals=[]
            for index,row in temp.iterrows():
                if((row[metric]<1)or(pd.isna(row[metric]))):
                    vals.append(1)
                elif(row[metric]<5):
                    vals.append(0.8)
                elif(row[metric]<10):
                    vals.append(0.4)
                else:
                    vals.append(0)
            temp[metric]=vals
    return temp
    
def Compute_DQ_Score(dataset_name_list,dataset_list,summarized_meta):
    metrics=list(summarized_meta.columns)
    metrics.remove('dataset_name')
    metrics.remove('column_name')
    metrics.remove('skewness')
    metrics.remove('skewness_level')
    temp=get_new_values(summarized_meta)
    scores={}
    for dataset_name,df in zip(dataset_name_list,dataset_list):
        weights=get_weights(dataset_name,summarized_meta)
        DQ_Score=[]
        for column in df.columns:
            value=0
            val=0
            for j,metric in enumerate(metrics):
                value=temp.loc[temp['column_name']==column,metric].values[0]*weights[j]
                if pd.isna(value):
                    val += weights[j]
                else:
                    val+=min(value,weights[j])
            DQ_Score.append(val)
        DQ_Score=np.array(DQ_Score)
        DQ_Score=np.mean(DQ_Score)*100
        scores[dataset_name]=float(DQ_Score)
    return scores
        


        
    
    
            
    



def Run_DQ_Pipeline(dataset_names,dataset_list,new_dataset_list=None):
    meta_table_combined=generate_meta(dataset_names,dataset_list,new_dataset_list) #step 1   
    recommendations=cleaning_recommendations(meta_table_combined)#step 2
    dataset_list_cleaned,change_log=get_cleaned_data(dataset_names,dataset_list,recommendations) #step 3             
    meta_table_combined_cleaned=generate_meta(dataset_names,dataset_list_cleaned)# step 4 optional  
    main_metrics=get_table_for_DQ_computation(meta_table_combined)#step 5
    main_metrics_cleaned=get_table_for_DQ_computation(meta_table_combined_cleaned)#step 6
    summarized_meta_table=summarize_dataset_health(meta_table_combined)#optional
    summarized_meta_table_cleaned=summarize_dataset_health(meta_table_combined_cleaned)#optional
    dirty_scores=Compute_DQ_Score(dataset_names,dataset_list,main_metrics)#step 7 
    cleaned_scores=Compute_DQ_Score(dataset_names,dataset_list_cleaned,main_metrics_cleaned)#step 8
    return { "dirty_scores": pd.DataFrame(dirty_scores.items(),columns=["Dataset_name", "DQ_Score"]), "cleaned_scores": pd.DataFrame(cleaned_scores.items(),columns=["Dataset_name", "DQ_Score"]),"cleaned_datasets":dataset_list_cleaned,"meta_table_before_cleaning": meta_table_combined, "meta_table_after_cleaning":meta_table_combined_cleaned, "recommendations": recommendations,"change_log": pd.DataFrame(change_log,columns=change_log[0].keys()), "summarized_meta_before_cleaning": summarized_meta_table, "summarized_meta_after_cleaning": summarized_meta_table_cleaned,"main_metrics_before_cleaning": main_metrics , "main_metrics_after_cleaning": main_metrics_cleaned }
result=Run_DQ_Pipeline(dataset_names,dataset_list)
result['dirty_scores']



result['meta_table_before_cleaning'].columns

