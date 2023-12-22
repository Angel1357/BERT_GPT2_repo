# Modelo BERT y GPT2 para la similitud de ofertas de trabajo

Este github contiene los codigos necesarios para poder obtener los embeddings de ofertas de trabajo, y realizar el descarte de ofertas que no superaron la similitud coseno deseada con los perfiles de trabajo.



# Como usar:

## Despues de clonar el repositorio:

### Instalar el enviroment `bert_gpt2_env_cpu` o `bert_gpt2_env_gpu` desde la ventana de comandos, despues de inicializar conda con:

```
python create_env.py -i BERT_GPT2_Env.yml.meta -f linux -f use_cpu
```
#### Si se desea instalar en windows, cambiar `linux` por `windows`, si se desea usar gpu, cambiar `use_cpu` por `use_gpu`, lo que crea el enviroment `bert_gpt2_env_gpu`.
```
conda activate bert_gpt2_env_cpu
```
## Bert_train o GPT2_train: 

Este script obtiene los datos a trabajar desde una base de datos a travez de sqlalchemy, y procede a entrenar el modelo Bert o GPT2 por las epocas definidas.

### Este spript requiere 2 flags:

`-env` que toma como argumento `dev` que se usa para identificar que credenciales extraer desde *credenciales.conf*.

`-use_cpu` o `-use_gpu` que especifica si se utilizara cpu o gpu al ejecutar el script, `-use_gpu` solo podra ejecutarse correctamente si se esta activado el enviroment `bert_env_gpu`.
```
python Bert_train.py -env dev -use_gpu
```

Linea 46 a 65 se utiliza sqlalchemy para obtener datos guardados en la base de datos de algun servidor, datos que se usaran para entrenar el modelo y se necesitara cambiar la direccion de donde se extraen de ser necesario, o obtener los datos con algun otro metodo.

Este codigo necesita los pesos de un modelo previamente entrenado en la misma carpeta de ejecucion, para asi poder continuar su entrenamiento, que se llama por su nombre en la linea 104, y en la linea 81 checkpoint_path define el nombre de la carpeta en la que se guardaran los pesos de los modelos entrenados al utlizar Bert_train.

La linea 78 contiene el parametro importante batch_size_num, que define el batch_size a ser utilizado en el entrenamiento, por defecto desde el metodo .fit este tiene un valor de 32, experimentalmente tiene que ser un maximo 3 en gpu 3070 laptop y maximo 8 en con gpu A4000.

La linea 79 contiene el parametro importante num_epochs, que define el numero de epocas que va a ser entrenado el modelo.

## Bert_prediction o GPT2_prediction:

Este script obtiene los datos a trabajar desde una base de datos a travez de sqlalchemy, y procede a realizar la prediccion de los embeddings con el modelo Bert, para su posterior calculo de similitud coseno. 

Al finalizar la rutina guarda un dataframe que contiene las ofertas de trabajo con su respectiva similitud coseno calculada.

### Este spript requiere 2 flags:

`-env` que toma como argumento `dev` que se usa para identificar que credenciales extraer desde *credenciales.conf*.

`-use_cpu` o `-use_gpu` que especifica si se utilizara cpu o gpu al ejecutar el script, `-use_gpu` solo podra ejecutarse correctamente si se esta activado el enviroment `bert_env_gpu`.
```
python Bert_prediction.py -env dev -use_gpu
```

Linea 46 a 65 se utiliza sqlalchemy para obtener datos guardados en la base de datos de algun servidor, datos que se usaran para entrenar el modelo y se necesitara cambiar la direccion de donde se extraen de ser necesario, o obtener los datos con algun otro metodo.

Este codigo necesita los pesos de un modelo previamente entrenado en la misma carpeta de ejecucion, para asi poder continuar su entrenamiento, que se llama por su nombre en la linea 94.

La linea 78 contiene el parametro importante batch_size_num, que define el batch_size a ser utilizado en el entrenamiento, por defecto desde el metodo .fit este tiene un valor de 32, experimentalmente tiene que ser un maximo 3 en gpu 3070 laptop y maximo 8 en con gpu A4000.

La linea 79 contiene el parametro importante predict_batch_num, que define cuantas predicciones se van a hacer antes de guardar los resultados, limpiar memoria ram y empezar con el siguiente lotem, experimentalmente un valor de 2000 ocupa un maximo de 16gb de ram y un valor de 1000 un maximo de 12gb de ram.

Linea 183 corresponde a el guardado de el que contiene las ofertas de trabajo con su respectiva similitud coseno calculada, si es que se desea guardar el dataframe que solo contiene los valores que estan por sobre el valor señalado, cambiar `df_ofertas_similitud` por `df_ofertas_similitud_muestreado`.
