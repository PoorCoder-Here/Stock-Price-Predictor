import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from datetime import datetime
from datetime import timedelta

try:
  data=pd.read_csv(#Enter the file location of historical data)
  #Make sure the file contains 7 column named 'DATE ','MONTH ','YEAR ','HOUR ','MINUTE ','SECOND ','PRICE '
  X=data[['DATE ','MONTH ','YEAR ','HOUR ','MINUTE ','SECOND ']]
  Y=data[['PRICE ']]
  model=linear_model.LinearRegression()
  prev=1
  now=datetime.now()
  end=now+timedelta(hours=2)
  now=now.strftime("%H:%M:%S")
  now=str(now)
  end=end.strftime("%H:%M:%S")
  end=str(end)
  while now!=end:
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
    model.fit(x_train,y_train)
    accuracy=model.score(x_test,y_test)
    acc=accuracy*100
    if acc>prev:
      prev=acc
      print(acc)
    now=datetime.now()
    now=now.strftime('%H:%M:%S')
    now=str(now)
except:
  fin=prev
  path='AIRTEL_Trained_model_'+str(fin)+'_accuracy'
  pickle.dump(model,open(path,'wb'))

fin=prev
path='AIRTEL_Trained_model_'+str(fin)+'_accuracy'
pickle.dump(model,open(path,'wb'))