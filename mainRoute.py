from flask import Flask,render_template,request
import pandas as pa
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

app=Flask(__name__)

@app.route('/')
def index():
	return render_template('homepage.html')
@app.route('/test')
def test():
	return render_template('test.html')
@app.route('/contact')
def contact():
	return render_template('contact.html')
@app.route('/aboutus')
def aboutus():
	return render_template('aboutus.html')
@app.route('/homepage')
def homepage():
	return render_template('homepage.html')
@app.route('/lung')
def lung():
	return render_template('lung.html')	


@app.route('/cancerPredict', methods=['POST'])
def cancerPredict():
    age=float(request.form['age'])
    gender=float(request.form['gender'])
    air=float(request.form['values'])
    alch=float(request.form['values1'])
    dust=float(request.form['values2'])
    occp=float(request.form['values3'])
    gene=float(request.form['values4'])
    ldesc=float(request.form['values5'])
    diet=float(request.form['values6'])
    obsty=float(request.form['values7'])
    smoke=float(request.form['values8'])
    psmoke=float(request.form['values9'])
    chest=float(request.form['values10'])
    cough=float(request.form['values11'])
    fatig=float(request.form['values12'])
    weight=float(request.form['values13'])
    breath=float(request.form['values14'])
    wheez=float(request.form['values15'])
    swallow=float(request.form['values16'])
    nails=float(request.form['values17'])
    cold=float(request.form['values18'])
    dcough=float(request.form['values19'])
    snore=float(request.form['values20'])
    data=pa.read_excel("cancer_patient_data_sets .xlsx").values
    #print(data)
    #print(data[0,1:24])
    train_data=data[0:998,1:24]
    train_target=data[0:998,24]
    '''print(train_target)
    test_data=data[999:,1:24]
    test_target=data[999:,24]
    print(test_target)'''
    clf=DecisionTreeClassifier()
    trained=clf.fit(train_data,train_target)
    clf1=SVC()
    trained1=clf1.fit(train_data,train_target)
    clf2=KNeighborsClassifier(n_neighbors=3)
    trained2=clf2.fit(train_data,train_target)

    test=[age,gender,air,alch,dust,occp,gene,ldesc,diet,obsty,smoke,psmoke,chest,cough,fatig,weight,breath,wheez,swallow,nails,cold,dcough,snore]
    #test=[34,1,2,3,4,5,6,7,6,5,4,3,2,1,2,3,4,5,2,3,5,2,3]
    predicted=trained.predict([test])
    predicted1=trained1.predict([test])
    predicted2=trained2.predict([test])

    print(predicted)
    print(predicted1)
    print(predicted2)

    #print(test_target)
    '''
    acc=accuracy_score(predicted,test_target)
    print(acc)
    acc1=accuracy_score(predicted1,test_target)
    print(acc)
    acc2=accuracy_score(predicted2,test_target)
    print(acc)
    '''
    #print(train_target)

    #print(age,gender,air,alch,dust,occp,gene,ldesc,diet,obsty,smoke,psmoke,chest,cough,fatig,weight,breath,wheez,swallow,nails,cold,dcough,snore)
    return render_template("lung.html",predicted=predicted,predicted1=predicted1,predicted2=predicted2)


if __name__ == '__main__':
    app.run(debug=True)
