import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = df=pd.read_csv('F.csv')
X=df.drop('Diagnosis',axis=1)
y=df['Diagnosis']
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42,stratify=y)
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

pickle.dump(model,open('model.pkl','wb'))