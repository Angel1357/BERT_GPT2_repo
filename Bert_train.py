import os
import configparser
import argparse

parser = argparse.ArgumentParser(prog='PROG',allow_abbrev=False)

help_env="Requiere especificar entorno de trabajo (dev, test ó prod)"
parser.add_argument("-env", "--ENVIRONMENT", help = help_env)

cpu_gpu = parser.add_mutually_exclusive_group(required=True)
cpu_gpu.add_argument('-use_cpu', action='store_true')
cpu_gpu.add_argument('-use_gpu', action='store_false')

args = parser.parse_args()
env = str(args.ENVIRONMENT)

cpu_gpu = args.use_cpu

config = configparser.ConfigParser()
config.read('credenciales.conf')

user=config['data_{}'.format(env)]['user']
password=config['data_{}'.format(env)]['password']
port=config['data_{}'.format(env)]['port']

if cpu_gpu==True:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import re
from unicodedata import normalize
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_nlp

import math
import sklearn
import math
import gc
os.environ["KERAS_BACKEND"] = "tensorflow"

import Bert_functions

#######################################################################

## importando los datos, hay 167113 al momento de crear este script

from sqlalchemy import create_engine
from sqlalchemy.sql import text


engine = create_engine("postgresql+psycopg2://"+user+":"+password+"@200.13.6.14:"+port+"/utilidades")
sql = '''SELECT concepto,categoria,cuerpo,modalidad,inclusividad FROM utils.ofertas_empleos;'''
with engine.connect().execution_options(autocommit=True) as conn:
    query = conn.execute(text(sql))
df1 = pd.DataFrame(query.fetchall())

engine = create_engine("postgresql+psycopg2://"+user+":"+password+"@200.13.6.14:"+port+"/utilidades")
sql = '''SELECT universidad,carrera,perfil_del_titulado,campo_laboral FROM utils.admision_ues_2022;'''
with engine.connect().execution_options(autocommit=True) as conn:
    query = conn.execute(text(sql))
df2 = pd.DataFrame(query.fetchall())


#######################################################################

## limpiando los datos

df1,df2,corpus=Bert_functions.preprocesador_2_corpus(df1,df2)


#######################################################################

from sklearn.model_selection import train_test_split
corpus_x, corpus_test = train_test_split(corpus, test_size=0.15, shuffle=True, random_state=13679)


#######################################################################

## Se carga el modelo base, y se le aplican los pesos de nuestro modelo previamente entrenado, cargar directamente el modelo no funciona

masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
    "bert_base_multi",
)

masked_lm.load_weights("./weights_2/practica2_weights_k")


#######################################################################

## batch_size = 32 is the default, maximo 3 en gpu 3070 laptop y max 8 en el servidor con gpu A4000

batch_size_now=8 # se define el batch_size
epoch_now=2 # se define el numero de epocas


#######################################################################

## Se evalua el modelo con 10 elementos aleatorios de el corpus

# corpus_test = pd.read_csv(r'corpus_test.csv',index_col=0)["cuerpo_pre"]
print(" ")
print("---------------------------------------------")
print("Evaluate")
print(" ")
masked_lm.evaluate(corpus.sample(100),batch_size=batch_size_now)
print(" ")

#######################################################################

# Entrenar el modelo

# Guarda los pesos en la carpeta training_weights, si no existe la crea, y guarda los pesos de la epoca en una carpeta con nombre cp-"epoca actual de el entrenamiento"
# Cambiar el nombre de la carpeta si se desea retomar entrenamiento, o va a sobre escribir modelo guardado

print("---------------------------------------------")
print("Entramiento del modelo")
print(" ")
checkpoint_path = "./training_weights/cp-{epoch:04d}/bert_weights" 
checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
   checkpoint_path, verbose=1, save_best_only=True, monitor="val_sparse_categorical_accuracy" ,save_weights_only=True,
   # Save weights when there is a improvement in the test accuracy
   #save_freq='epoch'
   )


history=masked_lm.fit(x=corpus_x,batch_size=batch_size_now,epochs=epoch_now,validation_data =corpus_test ,callbacks=[model_checkpoint_callback])

print(" ")
# Guarda la historia si se desea, se puede comentar si no
#np.save('my_history.npy',history.history)
#history_2=np.load('my_history.npy',allow_pickle='TRUE').item()
#history_2

