from flask import Flask,render_template,request,make_response
from flask import session
import pymysql as sql
import numpy as np
from iris import irismodel
app2=Flask(__name__)

@app2.route("/")
def index():
    return render_template("iris1.html")

@app2.route("/iris2/",methods=['GET','POST'])
def iris2():
    return render_template("iris2.html")

@app2.route("/afteriris2/",methods=['GET','POST'])
def afteriris2():
    B=""
    if request.method == 'POST':
        sl=float(request.form.get('sepal_length'))
        pl=float(request.form.get('petal_length'))
        sw=float(request.form.get('sepal_width'))
        pw=float(request.form.get('petal_width'))
        

        try:
          
            db = sql.connect(host="localhost",port=3306,user="root",password="",database="student")
        except Exception as e:
            return f"{e}"
        else:
            
            #c = db.cursor()
            #cmd = "insert into iris(sepallength,petallength,sepalwidth,petalwidth,species) values({},{},{},{},'{}')".format(sl,pl,sw,pw,B)
            #c.execute(cmd)
            #db.commit()

            model = irismodel()
            p=np.array([sl,pl,sw,pw])
            p=p.reshape(1,-1)
            A=model.predict(p)
        
            
            if 0<A<=0.5:
                B="setosa"
            elif 0.5<A<=1.5:
                B="versicolor"
            elif 1.5<A<=3:
                B="virginica"

            c = db.cursor()
            cmd = "insert into iris(sepallength,petallength,sepalwidth,petalwidth,species) values({},{},{},{},'{}')".format(sl,pl,sw,pw,B)
            c.execute(cmd)
            db.commit()    
    
            
            return render_template("afteriris2.html",data=B)
    else:
        return render_template("afteriris2.html")        

app2.run(debug=True)