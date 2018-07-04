import pandas as pa
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=pa.read_excel("cancer patient data sets.xlsx").values
#print(data)
#print(data[0,1:24])
train_data=data[0:998,1:24]
train_target=data[0:998,24]
#print(train_target)
test_data=data[999:,1:24]
test_target=data[999:,24]
#print('actual: ',test_target)

clf=DecisionTreeClassifier()
trained=clf.fit(train_data,train_target)
clf1=SVC()
trained1=clf1.fit(train_data,train_target)
clf2=KNeighborsClassifier(n_neighbors=3)
trained2=clf2.fit(train_data,train_target)
#x=input('enter:')
#x=input('enter:')x=input('enter:')x=input('enter:')x=input('enter:')x=input('enter:')x=input('enter:')x=input('enter:')
dict={'x':input('Enter age:'),'a':input('Enter1:'),'b':input('Enter2:'),'c':input('Enter3:'),'d':input('Enter4:'),'e':input('Enter5:')}
test=[[dict['x'],dict['a'],dict['b'],dict['c'],dict['d'],dict['e'],1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
predicted=trained.predict(test)
predicted1=trained1.predict(test)
predicted2=trained2.predict(test)

print(predicted)
print(predicted1)
print(predicted2)

#print(test_target)

acc=accuracy_score(predicted,test_target)
print(acc)
acc1=accuracy_score(predicted1,test_target)
print(acc1)
acc2=accuracy_score(predicted2,test_target)
print(acc2)
#print(train_target)
