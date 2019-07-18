from django.http import HttpResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status 
import json
import pickle
import numpy as np
import pandas as pd
import os
import psycopg2
import subprocess

train_file = os.path.abspath('train/train_model.py')
DIR_DATA = os.path.abspath('dataset/data_sans_prétraitement.csv')

@api_view(["POST"])
def store(request):
    try:
        body = json.loads(request.body)
        id = body["id"]
        text = body["text"]
        label = body["label"]
        
        #update
        connection = psycopg2.connect("dbname='data_emotions' user='postgres' password='123' host='localhost' port='5433'")
        mark = connection.cursor()
        statement_feedback = 'SELECT * FROM ' + 'feedback_emotion_store' + ' WHERE id = %s'
        mark.execute(statement_feedback,(id))
        ligne = mark.fetchone()
        if ligne[2] != label :
             statement_feedback = 'Update ' + 'feedback_emotion_store' + ' SET (label,validation) = (%s,%s) WHERE id = %s'
             mark.execute(statement_feedback,(label,0,id))
        statement_count = 'SELECT count(*) FROM feedback_emotion_store'
        mark.execute(statement_count)
        count = mark.fetchone()
        tab=[]
        if count[0] == 2:
             #sql = "COPY (SELECT text,label FROM feedback_sentiment_store) TO STDOUT WITH CSV DELIMITER ';'"
             #with open(DIR_DATA, "w") as file:
                #mark.copy_expert(sql, file)
             df = pd.read_csv(DIR_DATA, delimiter=';')
             row = 'SELECT text,label FROM feedback_emotion_store'
             mark.execute(row)
             row = mark.fetchone()
             while row is not None:
                  tab.append(row)
                  df.loc[len(df)]=[tab[0][1],tab[0][0]]
                  row = mark.fetchone()
             
             df.to_csv(DIR_DATA, sep=';', index=False)
             subprocess.run(['python ',train_file])
             #ch=""
             #file = 'C:/Users/amel/Desktop/test.sav'
             #pickle.dump(ch, open(file, 'wb'))

        connection.commit()
        connection.close()
        mark.close()

        return HttpResponse(json.dumps({"id":id,"message":text,"label":label}), content_type='application/json')
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)