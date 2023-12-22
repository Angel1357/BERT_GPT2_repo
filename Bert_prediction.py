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

## batch_size = 32 is the default, maximo 3 en gpu 3070 laptop y max 8 en el servidor con gpu A4000

batch_size_num=8 # se define el batch_size
predict_batch_num=1000 # se define cuantas predicciones se van a hacer antes de guardar los resultados, limpiar memoria ram y empezar con el siguiente lote


#######################################################################
#######################################################################
#######################################################################
#######################################################################

## Se carga el modelo base, y se le aplican los pesos de nuestro modelo previamente entrenado, cargar directamente el modelo no funciona

masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
    "bert_base_multi",
)


masked_lm.load_weights("./weights_bert_10_epoch/bert_weights")

predict_lm=masked_lm.backbone # bert.backbone es lo que realiza la prediccion de embeddings
preprocessor_get_word_embedding = keras_nlp.models.BertPreprocessor.from_preset("bert_base_multi") # preprocesador necesario para usar bert.backbone


#######################################################################

Bert_functions.predict_lm=predict_lm
Bert_functions.preprocessor_get_word_embedding = preprocessor_get_word_embedding
Bert_functions.df1 = df1
Bert_functions.df2 = df2


#######################################################################

## Se evalua el modelo con 10 elementos aleatorios de el corpus

# corpus_test = pd.read_csv(r'corpus_test.csv',index_col=0)["cuerpo_pre"]
print(" ")
print("---------------------------------------------")
print("Evaluate")
print(" ")
masked_lm.evaluate(corpus.sample(100),batch_size=batch_size_num)
print(" ")


#######################################################################

# Se obtienen vectores por texto de las ofertas de trabajo en df1
# obtiene para todo df1, si no se desea se puede señalar que posicion obtener, ejemplo df1[0:100]
print("---------------------------------------------")
print("df1 prediction")
print(" ")
df1_centroid,df1_matched=Bert_functions.centroid_df(df1,columns=["concepto","cuerpo_pre"],predict_batch=predict_batch_num,batch_size=batch_size_num)


## Funciones para guardar y cargar los datos de ser necesario
## Cambiar nombre de ser necesario para no sobre escribir un archivo antiguo

# df1_matched.to_pickle("df1_matched.pkl")
# df1_matched = pd.read_pickle("df1_matched.pkl")
print(" ")


#######################################################################

## Se obtienen vectores por texto de las carreras
print("---------------------------------------------")
print("df2 prediction")
print(" ")
df2_centroid,df2_matched_centroid=Bert_functions.centroid_df(df2,columns=["universidad","carrera","cuerpo_pre"],predict_batch=predict_batch_num,batch_size=batch_size_num)

## Funciones para guardar y cargar los datos de ser necesario
## Cambiar nombre de ser necesario para no sobre escribir un archivo antiguo

# df2_matched_centroid.to_pickle("df2_matched.pkl")
# df2_matched_centroid = pd.read_pickle("df2_matched.pkl")
print(" ")


#######################################################################

## Funciones usadas para visualizar porcentaje de datos que quedan con distintos valores de corte

valor_corte_similitud=0.70

print("---------------------------------------------")
df_ofertas_similitud,df_ofertas_similitud_muestreado,df_porcentajes=Bert_functions.get_all_distances(valor_corte_similitud,df2_matched_centroid,df1_matched,0,min_max=0,cut_porcentaje_por_carrera=False)
per_list=Bert_functions.valores_corte_porc(df_ofertas_similitud,60,100)


#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################


## Funciones para guardar y cargar los datos de ser necesario
## Cambiar nombre de ser necesario para no sobre escribir un archivo antiguo

## Facilmente se puede obtener el df muestrado desde el completo la siguiente forma
## df_ofertas_similitud[df_ofertas_similitud.distancia>=0.7], siendo este un datagrame que toma solo los datos
## con similitud coseno mayor a 0.7

df_ofertas_similitud.to_pickle("df_ofertas_similitud_bert.pkl")
# df_ofertas_similitud = pd.read_pickle("df_ofertas_similitud.pkl")




