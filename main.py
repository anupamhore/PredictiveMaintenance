from flask import Flask, request,render_template,Response
from flask_cors import CORS, cross_origin
from trainingmodel import trainmodel
from predictionmodel import predictmodel
import os

os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predictairpressure",methods=['POST'])
@cross_origin()
def predictAirPressure():

    try:
        if request.method == 'POST':
            type = request.form['type']
            machine_failure = request.form['machine_failure']
            twf = request.form['twf']
            hdf = request.form['hdf']
            pwf = request.form['pwf']
            osf = request.form['osf']
            rnf = request.form['rnf']

            if request.form['process_temp'] != "":
                process_temp = float(request.form['process_temp'])
            else:
                return render_template('index.html', result='Please input valid process temperature', result1='')
            if request.form['rotational_speed'] != "":
                rotational_speed = int(request.form['rotational_speed'])
            else:
                return render_template('index.html', result='Please input valid rotational speed', result1='')

            if request.form['torque'] != "":
                torque = float(request.form['torque'])
            else:
                return render_template('index.html', result='Please input valid torque', result1='')

            if request.form['toolwear'] != "":
                toolwear = float(request.form['toolwear'])
            else:
                return render_template('index.html', result='Please input valid tool wear ', result1='')

            if type == 'Select' or machine_failure == 'Select' or twf == 'Select' or hdf == 'Select' or pwf == 'Select' or osf == 'Select' or rnf == 'Select' :
                return render_template('index.html', result='Please input valid values', result1='')

            if process_temp is None or rotational_speed is None or torque is None or toolwear is None:
                return render_template('index.html', result='Please input valid values', result1='')

            predictObj = {'type':type,'process_temp':process_temp,'rotational_speed':rotational_speed,
                          'torque':torque,'toolwear':toolwear,'machineFail':machine_failure,'twf':twf,
                          'hdf':hdf,'pwf':pwf,'osf':osf,'rnf':rnf}

            predictionmodel_obj = predictmodel(predictObj, 'path_to_file')
            result = predictionmodel_obj.predict()
            result = float("{0:.2f}".format(result))
            response = 'Air Temperature:' + str(result) + 'K'
            response1 = "Type:{} \n Pressure Temp:{} \n Rotational Speed:{} \n Torque:{} \n ToolWear:{} \n Machine Failure:{} \n TWF:{} \n HDF:{} \n PWF:{} \n OSF:{} \n RNF:{} \n".format(str(type),str(process_temp),str(rotational_speed),str(torque),str(toolwear), str(machine_failure),str(twf),str(hdf),str(pwf),str(osf),str(rnf))

            return render_template('index.html', result=response,result1=response1)
    except Exception as e:
        raise Exception(e)

@app.route("/predict",methods=['POST'])
@cross_origin()
def predictRouteClient():

    try:
        if request.json is not None:
            type = request.json['type']
            process_temp = request.json['process_temp']
            rotational_speed = request.json['rotational_speed']
            torque = request.json['torque']
            toolwear = request.json['toolwear']
            machineFail = request.json['machineFail']
            twf = request.json['twf']
            hdf = request.json['hdf']
            pwf = request.json['pwf']
            osf = request.json['osf']
            rnf = request.json['rnf']

            predictObj = {'type':type,'process_temp':process_temp,'rotational_speed':rotational_speed,
                          'torque':torque,'toolwear':toolwear,'machineFail':machineFail,'twf':twf,
                          'hdf':hdf,'pwf':pwf,'osf':osf,'rnf':rnf}

            predictionmodel_obj = predictmodel(predictObj, 'path_to_file')
            result = predictionmodel_obj.predict()
            print('Air Temperature:{}'.format(result))


    except Exception as e:
        raise Exception(e)

    return Response("Prediction successfull!!")

@app.route("/predictmass",methods=['POST'])
@cross_origin()
def predictMass():

    try:
        if request.json['filepath'] is not None:
            predictObj = {}
            predictionmodel_obj = predictmodel(predictObj, 'path_to_file')
            predictionmodel_obj.predictMass()


    except Exception as e:
        raise Exception(e)

    return Response("Prediction successfull!!")


@app.route("/train",methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['filepath'] is not None:
            path = request.json['filepath']
            trainmodel_obj = trainmodel(path)
            trainmodel_obj.trainingModel()


    except Exception as e:
        print("Exception: " + str(e))
        return Response("Training Error!!" + str(e))

    return Response("Training successfull!!")


if __name__ == "__main__":
    app.run(debug=False)
