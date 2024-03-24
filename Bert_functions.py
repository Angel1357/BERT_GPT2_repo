import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


## limpiando los datos

def preprocesador(texto):
    """
    Este preprocesador lleva las letras a minúscula
    reemplaza por espacio un grupo de caracteres,
    elimina los dígitos numéricos y las tíldes.
    Finalmente, deja todas las palabras separadas por un solo espacio.
    
    """
    puntuacion = r'[,;.:•“”¡!¿?<>@#$%&[\](){}<>~=+\-*/|\\_^`"\']'
    texto = re.sub(puntuacion, ' ', texto)
    texto = re.sub(r'\d','', texto)
    texto = texto.lower()
    texto = re.sub('á', 'a', texto)
    texto = re.sub('é', 'e', texto)
    texto = re.sub('í', 'i', texto)
    texto = re.sub('ó', 'o', texto)
    texto = re.sub('ú', 'u', texto)
    texto = re.sub('ü', 'u', texto)
    #texto = re.sub('ñ', 'n', texto)
    texto = ' '.join(re.findall(r"[,/.\-:()\w]+", texto))
    
    return texto


def preprocesador_2_corpus(df1,df2):
    df1['cuerpo_pre'] = df1['cuerpo'].apply(preprocesador)
    df1 = df1.drop_duplicates(subset=['cuerpo']).dropna(subset=['cuerpo'])

    df2['carrera'] = df2['carrera'].apply(preprocesador)

    ## creando el corpus de entrenamiento

    lista = []
    for i in range(len(df2)):
        if type(df2['campo_laboral'].iloc[i]) == str:
            if type(df2['perfil_del_titulado'].iloc[i]) == str:
                texto = df2['perfil_del_titulado'].iloc[i] + ' ' + df2['campo_laboral'].iloc[i]
                
            else: 
                texto = df2['campo_laboral'].iloc[i]
            
        elif type(df2['perfil_del_titulado'].iloc[i]) == str:
            texto = df2['perfil_del_titulado'].iloc[i]

        else:
            texto = df2['perfil_del_titulado'].iloc[i]
        lista.append(texto)
        
    df2['cuerpo'] = lista
    df2 = df2.dropna(subset=['cuerpo'])
    df2['cuerpo_pre'] = df2['cuerpo'].apply(preprocesador)
    df2 = df2.drop_duplicates(subset=['cuerpo_pre'])

    corpus = pd.concat([df2['cuerpo_pre'],df1['cuerpo_pre']]).reset_index(drop=True).copy()

    # Se revisa si hay vacios en cuerpo_pre
    print(" ")
    print("-----------------------")         
    print("Number of missing values in df1")
    print(df1.isnull().sum())

    print(" ")
    print("-----------------------")
    print(" ")
    # Se revisa si hay vacios en cuerpo_pre           
    print("Number of missing values in df2")
    print(df2.isnull().sum())

    print(" ")
    print("-----------------------")
    print(" ")
    return df1,df2,corpus


def get_word_embedding(text,predict_batch,batch_size_n):

    """
    Funcion que obtiene los embeddings de un corpus a travez de el metodo predict() de bert.backbone

    Inputs:

    text: corpus a predecir embeddings
    predict_batch: cantidad de datos que se van a usar de una sola pasada en el metodo .predict()
    batch_size_n: batch_size para el .metodo predict()
    
    Outputs:

    output_list: contiene outputs['sequence_output'] y outputs['pooled_output']

    outputs['sequence_output']: Contiene los emb por token de la oracion, 512x768, 512 tokens, 768 valores para cada token, para cada texto en text
    outputs['pooled_output']: is the mean pooling of all hidden states, contiene 768 valores que representan los 512 tokens de la oracion,
                                    para cada texto en text

    """
    output_list=[[],[]]
    

    input_ids = preprocessor_get_word_embedding([text])



    for i in range(math.ceil(len(input_ids['token_ids'])/predict_batch)):

        tf.keras.backend.clear_session()
        gc.collect()
        batch_input_ids = {k: input_ids[k][i*predict_batch:(i+1)*predict_batch] for k in input_ids.keys()}
        outputs = predict_lm.predict(batch_input_ids,batch_size=batch_size_n)
        
        tf.keras.backend.clear_session()
        gc.collect()

        output_list[0].extend(outputs['sequence_output'])
        output_list[1].extend(outputs['pooled_output'])
        

        
    # outputs['sequence_output'] is token vector
    # outputs['pooled_output'] is the mean pooling of all hidden states
    return output_list
    


def get_centroid_emb(corpus,predict_batch,batch_size_n):

    """
    Funcion que divide un corpus en partes mas pequeñas que pasar a la funcion get_word_embedding(), y calcula embeddings por oracion

    Inputs:

    corpus: corpus a predecir embeddings
    predict_batch: cantidad de datos que se van a usar de una sola pasada en el metodo .predict()
    batch_size_n: batch_size para el .metodo predict()
    
    Outputs:

    centroid_emb_list: np.mean() con axis 0, pasa outputs['sequence_output'] a vectores de largo 768
    centroid_emb_list_1: np.mean() con axis 1, pasa outputs['sequence_output'] a vectores de largo 512
    cls_tokens: outputs['pooled_output']

    outputs['pooled_output']: is the mean pooling of all hidden states, contiene 768 valores que representan los 512 tokens de la oracion,
                                para cada texto en corpus

    """
    
    centroid_emb_list=[]
    centroid_emb_list_1=[]
    cls_tokens=[]
    for i in range(math.ceil(len(corpus)/predict_batch)):

        print("---------------------------------------------")
        print('Prediccion de datos entre  '+str(predict_batch*i)+" : "+str(predict_batch*(i+1)))
        print(" ")
        gc.collect()
        corpus_vec=get_word_embedding(corpus[i*predict_batch:(i+1)*predict_batch],predict_batch,batch_size_n)
        
        for vector in corpus_vec[0]:
            # axis 0 pasa a vectores de largo 768
            centroid_emb=np.mean(vector,axis=0)
            centroid_emb_list.append(centroid_emb) 
 
            # axis 1 pasa a vectores de largo 512, 1 valor por cada palabra
            centroid_emb_1=np.mean(vector,axis=1) 
            centroid_emb_list_1.append(centroid_emb_1)

        cls_tokens.extend(corpus_vec[1])

    return centroid_emb_list,centroid_emb_list_1, cls_tokens

def centroid_df(df,predict_batch,batch_size,columns=[]):
    """
    Funcion que aplica la funcion get_centroid_emb para la obtencion de embeddings, y filtra solo por solo los datos que tengan palabra clave existentes en df1 y df2
    y los guarda en df_matched con la funcion matched_df

    Inputs:

    df: dataframe que contiene 4 columnas, carrera, cuerpo_pre, distancia (la distancia coseno para cada dato), desviacion (desviacion para cada dato)
    columns: Las columnas a guardar de df en df_matched
    predict_batch: La cantidad de datos a usar simultaneamente en el metodo .predict()
    batch_size: el valor de batch_size para el metodo .predict()

    
    Outputs:
    df_to_centroid: Dataframe que contiene los datos de df1 elegidos en columns, junto con los embbedings de los textos en cuerpo_pre
    df_matched: Dataframe que contiene solo los datos que tienen palabra clave existentes simultanemante en df1 y df2
    

    """
    #df_to_centroid=df[columns].copy()
    df_to_centroid=df.copy()
    df_centroid=get_centroid_emb(df_to_centroid.cuerpo_pre,predict_batch,batch_size)
    df_to_centroid['0']=df_centroid[0] #768
    #df_to_centroid['1']=df_centroid[1] #512
    #df_to_centroid['2']=df_centroid[2] #cls



    df_matched=matched_df(df_to_centroid)

    return df_to_centroid,df_matched

def matched_df(df_to_centroid):
    """
    Funcion que filtra solo por solo los datos que tengan palabra clave existentes en df1 y df2
    y los guarda en df_matched

    Inputs:

    df_to_centroid: Dataframe que contiene los datos de df1 elegidos en columns, junto con los embbedings de los textos en cuerpo_pre
    
    Outputs:
    
    df_matched: Dataframe que contiene solo los datos que tienen palabra clave existentes simultanemante en df1 y df2 
    
    """

    if np.isin("carrera",df_to_centroid.columns):
        # Obtengo la lista de conceptos/carreras que estan en ambos dataframes
        concepto_carrera_bolean=np.isin(df1.concepto.unique(), df2.carrera.unique())
        only_matched_concepts=[ df1.concepto.unique()[i]  for i in range(len(df1.concepto.unique())) if concepto_carrera_bolean[i]==True]

        df_matched=df_to_centroid[df_to_centroid.carrera.isin(only_matched_concepts)].copy()

    elif np.isin("concepto",df_to_centroid.columns):
        # Obtengo la lista de conceptos/carreras que estan en ambos dataframes
        concepto_carrera_bolean=np.isin(df_to_centroid.concepto.unique(), df2.carrera.unique())
        only_matched_concepts=[ df_to_centroid.concepto.unique()[i]  for i in range(len(df_to_centroid.concepto.unique())) if concepto_carrera_bolean[i]==True]

        df_matched=df_to_centroid[df_to_centroid.concepto.isin(only_matched_concepts)].copy()

    else:
        print("No se pudo encontrar la columna con palabra clave, carrera o concepto, por lo que se devuelve todo el dataframe y no solo las coincidencias")
        df_matched=df_to_centroid.copy()

    return df_matched

# Similitud coseno
def sim_coseno(vec1, vec2):
    cos = np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos

def get_distances(cut,X1_input,X2_input,n_centroid,min_max_bolean,cut_porcentaje_por_carrera=False):
    """
    Funcion que obtiene las distancias entre dos dataframes de un solo concepto

    Inputs:

    cut: Valor de corte para la similitud coseno, datos con valores menores no se guardan en df1_dist_sampled
    X1_input: dataframe con almenos 3 columnas; carrera, cuerpo_pre, y un valor str como nombre de la columna donde se encuentran los emb 
    X2_input: dataframe con almenos 3 columnas; concepto, cuerpo_pre, y un valor str como nombre de la columna donde se encuentran los emb
    n_centroid: nombre de la columna en donde se encuentra los emb
    min_max_bolean: 1 si se aplica minmax_scale, 0 si no se aplica
    cut_porcentaje_por_carrera: si Verdadero, se corta para que solo quede un porcetaje superior a cut por carrera
    
    Outputs:

    df1_dist: dataframe que contiene 4 columnas, carrera, cuerpo_pre, distancia (la distancia coseno para cada dato), desviacion (desviacion para cada dato)
    df1_dist_sampled: dataframe que contiene los datos de df1_dist que cumple la condicion de ser mayor que valor de corte "cut"

    """

    X1 = X1_input[str(n_centroid)].to_list() # X1 contiene todos los emb de las carreras
    X2 = X2_input[str(n_centroid)].to_list() # X1 contiene todos los emb de las ofertas de trabajo


    if len(X2)!=0:
        if min_max_bolean==1:
            X3=X1.copy()
            X3.extend(X2)
            X3=sklearn.preprocessing.minmax_scale(X3)

            #std_esc=StandardScaler()
            #std_fit=std_esc.fit(X3)
            #X3=std_esc.transform(X3)

            X2=X3[len(X1):]
            X1=X3[:len(X1)]

       
    dist_prom = []
    desv_std = []
    for i in range(len(X2)):
        dists = []
        for j in range(len(X1)):
            dist = sim_coseno(X1[j],X2[i])

            dists.append(dist)
        dist_prom.append(np.mean(dists))
        desv_std.append(np.std(dists))


    df1_dist=X2_input.copy()
    df1_dist['distancia'] = dist_prom
    df1_dist['desviacion'] = desv_std


    df1_dist_sampled = df1_dist.loc[(df1_dist.loc[:,'distancia'] >= cut)].copy()

    if cut_porcentaje_por_carrera==True:

        per_list,iter_n=valores_corte_porc(df1_dist,0,100,0.1,print_bolean=False)
        per_listdf=pd.DataFrame([per_list,iter_n]).T
        per_listdf_sample=per_listdf.loc[(per_listdf.loc[:,0] >= cut*100)].copy()
        
        new_cut=per_listdf_sample[1][len(per_listdf_sample)-1]
        df1_dist_sampled = df1_dist.loc[(df1_dist.loc[:,'distancia'] >= new_cut)].copy()

    return df1_dist, df1_dist_sampled


 

def get_all_distances(cut,df2_matched_centroid,df1_matched, n_centroid, min_max,cut_porcentaje_por_carrera=False):

    """
    Funcion que divide el dataframe general en dataframes por carrera/concepto, y usa la funcion get_distances para obtener las distancias

    Inputs:

    cut: Valor de corte para la similitud coseno, datos con valores menores no se guardan en df1_dist_sampled

    Estas dataframes son solo con los conceptos/carreras que existen simultaneamente en ambos dataframes 
    df2_matched_centroid: dataframe con almenos 3 columnas; carrera, cuerpo_pre, y un valor str como nombre de la columna donde se encuentran los emb 
    df1_matched: dataframe con almenos 3 columnas; concepto, cuerpo_pre, y un valor str como nombre de la columna donde se encuentran los emb
    cut_porcentaje_por_carrera: si Verdadero, se corta para que solo quede un porcetaje superior a cut por carrera


    n_centroid: nombre de la columna en donde se encuentra los emb
    min_max: 1 si se aplica minmax_scale por carrera, 2 si se aplica a todo el dataframe, 0 si no se aplica
    
    Outputs:

    df_end: dataframe que contiene 4 columnas, carrera, cuerpo_pre, distancia (la distancia coseno para cada dato), desviacion (desviacion para cada dato)
    df_end_sampled: dataframe que contiene los datos de df1_dist que cumple la condicion de ser mayor que valor de corte "cut"
    df_from_to: un dataframe que contiene como indice los conceptos/carreras, y como columnas la cantidad de datos iniciales antes del corte
                , y la cantidad de datos restantes despues despues del corte 

    """


    df_end=pd.DataFrame()
    df_end_sampled=pd.DataFrame()
    df_from_to=pd.DataFrame()

    if min_max==2:
        X3=df2_matched_centroid["0"].copy().to_list()
        X3.extend(df1_matched["0"].copy().to_list())
        X3=sklearn.preprocessing.minmax_scale(X3)

        X2=X3[len(df2_matched_centroid):]
        X1=X3[:len(df2_matched_centroid)]

        df2_matched_centroid=df2_matched_centroid.copy()
        df2_matched_centroid["0"]=list(X1)
        df1_matched=df1_matched.copy()
        df1_matched["0"]=list(X2)

   

    
    # este for divide el dataframe general en dataframes por carrera, para ser enviados a la funcion get_distances
    for carrera_for in df1_matched.concepto.unique():

        #Dataframes por carrera
        centroids_df2_carrera=df2_matched_centroid[df2_matched_centroid.carrera==carrera_for].copy()
        df1_carrera=df1_matched[df1_matched.concepto==carrera_for].copy()

        #Se obtienen las distancias
        df1_dist, df1_dist_sampled=get_distances(cut,centroids_df2_carrera,df1_carrera,n_centroid,min_max,cut_porcentaje_por_carrera)

        #Se almacenan para retornarlas
        df_end=pd.concat([df_end,df1_dist])
        df_end_sampled=pd.concat([df_end_sampled,df1_dist_sampled])
        df_from_to[carrera_for]=[len(df1_dist),len(df1_dist_sampled),(len(df1_dist_sampled)/len(df1_dist))*100]
        
    df_end=df_end.sort_index(ascending=True)
    df_end_sampled=df_end_sampled.sort_index(ascending=True)
    
    porcent_colum=str((df_from_to.T[1].sum()/df_from_to.T[0].sum())*100)[0:5]+" %"
    df_from_to=df_from_to.set_axis(["Inicial", "Final",porcent_colum], axis="index").T

    
    return df_end,df_end_sampled,df_from_to


def valores_corte_porc(df1_end_fin,inicio,final,step=1,print_bolean=True):

    """
    Funcion que toma el DataFrame con las distancias calculadas, y aplica distintos valores de corte para calcular porcentaje de datos restantes

    Inputs:

    df_end: dataframe que contiene 4 columnas, carrera, cuerpo_pre, distancia (la distancia coseno para cada dato), desviacion (desviacion para cada dato)
    inicio: el inicio para la funcion range(inicio,final)
    final: el final para la funcion range(inicio,final)
    step: El paso entre cada valor entre inicio y final
    print_bolean: Si False entonces no va a usar la funcion print() para mostrar resultados en pantalla
    

    
    Outputs:

    iter_n: lista que guarda los valores de corte probados
    iter_n_value: lista que guarda los valores de porcentaje restante al aplicar los valores de corte
    

    """

    # For usado para visualizar porcentaje de datos que quedan con distintos valores de corte
    if print_bolean==True:
        print("---------------------------------------------")
        print("| Similitud corte   |   porcentaje restante |")
        print("|-------------------------------------------|")
    iter_n=[]
    iter_n_value=[]
    xs = (x * step for x in range(int(inicio*((step**-1))), int(final*((step**-1)))))
   
    for i in xs:
        id=i/100

        
        df1_dist_sampled = df1_end_fin.loc[(df1_end_fin.loc[:,'distancia'] >= id)].copy()

        #print("Similitud corte: "+str(i)+"%,    porcentaje restante "+str((len(df1_dist_sampled)/len(df1_end_fin))*100)[0:7]+"%")
        if print_bolean==True:
            print("|         "+str(id*100)[0:4]+"%       -   "+str((len(df1_dist_sampled)/len(df1_end_fin))*100)[0:7]+"%          |")

        iter_n.append(id)
        iter_n_value.append((len(df1_dist_sampled)/len(df1_end_fin))*100)
        
    return iter_n_value,iter_n

def sampling_df(df1_end_fin,inicio,final):

    """
    Funcion que toma una muestra de el DataFrame con las distancias calculadas, siendo el nuevo dataframe solo los datos con distancias entre inicio y final

    Inputs:

    df_end: dataframe que contiene 4 columnas, carrera, cuerpo_pre, distancia (la distancia coseno para cada dato), desviacion (desviacion para cada dato)
    inicio: el valor de corte inferior
    final: el valor de corte superior

    
    Outputs:

    df1_dist_sampled: Dataframe que contiene solo los datos con valores de corte entre inicio y final
    

    """

    df1_dist_sampled = df1_end_fin.loc[(df1_end_fin.loc[:,'distancia'] >= inicio)].copy()
    df1_dist_sampled = df1_dist_sampled.loc[(df1_dist_sampled.loc[:,'distancia'] <= final)].copy()
    
    return df1_dist_sampled

def random_sampling_df(df,inicio,final,step,n_samples=1):
        
    """
    Funcion que toma muestras de el DataFrame con las distancias calculadas, tomando un 1 valor por cada intervalo formado por i*step/i*step+step

    Inputs:

    df_end: dataframe que contiene 4 columnas, carrera, cuerpo_pre, distancia (la distancia coseno para cada dato), desviacion (desviacion para cada dato)
    inicio: el valor de corte inferior
    final: el valor de corte superior
    step: valor de paso para los intervalos

    
    Outputs:

    five_sampled_df: Dataframe que contiene 1 muestra por cada intervalo dado por step
    """

        #per_list,iter_n=valores_corte_porc(df,0,100,0.1,print_bolean=False)
        #per_listdf=pd.DataFrame([per_list,iter_n]).T

        #n_=100/n_
    xs = (int((x * step)*100)/100 for x in range(int(inicio*((step**-1))), int(final*((step**-1)))))
    five_sampled_df=pd.DataFrame()
    for i in xs:
            id= int(i*10)/1000
            #print(i)
            id_next=int(((i+step)*10))/1000
            #print(id_next)
            #print(""+str(id)+" - "+str(id_next))
                
            if sampling_df(df,id,id_next).empty:
                        #print("1")
                    continue
            else:
                        #print("2")
                    five_sampled_df=pd.concat([five_sampled_df.copy(),sampling_df(df,id,id_next).sort_values("distancia").sample(n=n_samples)]).copy()

    return five_sampled_df




