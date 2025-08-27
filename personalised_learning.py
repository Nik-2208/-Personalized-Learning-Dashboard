import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv(r"C:\Users\nikhi\OneDrive\Desktop\Personalized Learning\archive\student_performance_large_dataset.csv")

df['Gender_en'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['Participation_in_Discussions_en'] = df['Participation_in_Discussions'].map({'No':0, 'Yes':1})
df['Use_of_Educational_Tech_en'] = df['Use_of_Educational_Tech'].map({'No':0, 'Yes':1})
df['Self_Reported_Stress_Level_en'] = df['Self_Reported_Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Final_Grade_en'] = df['Final_Grade'].map({'A':5, 'B':4, 'C':3, 'D':2, 'E':1, 'F':0})

# print(df.isna().sum())

oe = OneHotEncoder(sparse_output=False, drop='first')
encoded = oe.fit_transform(df[['Preferred_Learning_Style']])
encoded_df = pd.DataFrame(encoded, columns=oe.get_feature_names_out(['Preferred_Learning_Style']))
df = pd.concat([df,encoded_df],axis=1)

ss = StandardScaler()
df[['Age','Study_Hours_per_Week']]= ss.fit_transform(df[['Age','Study_Hours_per_Week']])

df.drop(['Student_ID','Online_Courses_Completed','Assignment_Completion_Rate (%)', 'Exam_Score (%)'],axis=1)

X = df[['Age', 'Study_Hours_per_Week', 'Participation_in_Discussions_en', 'Attendance_Rate (%)', 'Self_Reported_Stress_Level_en', 'Time_Spent_on_Social_Media (hours/week)','Sleep_Hours_per_Night', 'Gender_en','Preferred_Learning_Style_Kinesthetic','Preferred_Learning_Style_Reading/Writing','Preferred_Learning_Style_Visual']]
y = df['Final_Grade_en']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)

# en = ElasticNet(alpha=1,l1_ratio=0.5,random_state=42)
# en.fit(X_train, y_train)
# print('Coef: ',en.coef_)
# print('Intercept: ',en.intercept_)

rfr = RandomForestRegressor(n_estimators=100,max_depth=20)
rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test)

# print('MAE: ', mean_absolute_error(y_test,y_pred))
# print('MSE: ', mean_squared_error(y_test,y_pred))
# print('RMSE: ', root_mean_squared_error(y_test,y_pred))
# print('R2: ', r2_score(y_test,y_pred))
# print(df.select_dtypes(include=['number']).corr())

joblib.dump(rfr, "rf_model.pkl", compress=3)  
joblib.dump(ss, "scaler.pkl", compress=2)      
joblib.dump(oe, "encoder.pkl", compress=2)     

print("Model training complete and saved")
