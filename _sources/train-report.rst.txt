Train Report
==========================

Generación de informes PDF a partir de ficheros CSV con los datos de los acelerómetros, agrupando lógicamente los eventos de vibración detectados para determinar qué vibraciones corresponden al paso de los trenes y su posterior resumen en un fichero de reportes.

Descripción
-----------

El script ``train-report-smartbridges.py`` procesa los archivos CSV de los datos en crudo para cada uno de los sensores, agrupando los eventos de trenes detectados en función de la coincidencia temporal entre sensores.

Una vez determinadas qué vibraciones corresponden a un tren, se genera una página en un informe PDF con gráficas de aceleración por cada eje y la transformada FFT.

Para una explicación más detallada del proceso, véase 

Requisitos
----------

- Python 3.7+
- Librerías externas (versiones mínimas recomendadas):

  - `matplotlib <https://matplotlib.org/>`_ >= 3.5
  - `numpy <https://numpy.org/>`_ >= 1.21
  - `pandas <https://pandas.pydata.org/>`_ >= 1.3
  - `seaborn <https://seaborn.pydata.org/>`_ >= 0.11
  - `scipy <https://scipy.org/>`_ >= 1.7
  - `PyPDF2 <https://pypdf2.readthedocs.io/>`_ >= 2.0

Instalación de dependencias
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Para instalar las dependencias necesarias, asegúrate de tener `pip` instalado y ejecuta el siguiente comando:

.. code-block:: bash

   pip install -r requirements.txt


Uso
---

El script se ejecuta desde la línea de comandos y requiere la ruta al directorio del puente donde se encuentran los datos de los sensores. El script buscará en la carpeta `raw` dentro de esa ruta los archivos CSV con los datos de aceleración.

Para ejecutar el script, utiliza el siguiente comando:

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente

El script tiene dos modos de operación:

1. **Generación de informes**: Procesa los datos de los sensores y genera un informe PDF con las gráficas de aceleración y FFT.
2. **Regeneración de páginas específicas**: Permite regenerar páginas concretas del informe en una fecha y horas específicas.


Argumentos
~~~~~~~~~~

El script ``train-report-smartbridges.py`` acepta los siguientes argumentos:

- ``--bridge_path``: Ruta a la carpeta del puente donde se encuentran los datos.

Parámetros opcionales:

- ``--date`` (opcional) : Fecha a procesar en formato ``YYYYMMDD``. Si no se indica, usa el día anterior al día del sistema.
- ``--version``         : Muestra la versión del programa y termina la ejecución.
- ``--min_sensors N``   : Número mínimo de sensores para considerar un evento (por defecto: 5)
- ``--workers N``       : Número de hilos para el procesamiento paralelo (por defecto: 2 x núcleos CPU)
- ``--regenerar-hora``  : Una o varias horas (formato ``HH:MM:SS``) para regenerar páginas concretas del informe.

En la siguiente sección se detallan ejemplos de uso del script.

Ejemplos de uso
-----------------

Creación de informes
~~~~~~~~~~~~~~~~~~~~

Para generar el informe completo del día anterior:

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente


Para generar el informe especificando el número de sensores mínimos y el número de hilos:

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente --min_sensors 3 --workers 4

Para generar el informe de una fecha concreta (formato YYYYMMDD):

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente --date 20240601


Regenerar páginas específicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Si necesitas regenerar la página de un tren concreto (por ejemplo, el tren detectado a las 12:34:56):

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente --date 20240601 --regenerar-hora 12:34:56

Regenear varias páginas para diferentes horas:

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente --date 20240601 --regenerar-hora 12:34:56 13:45:00

Para regenerar alguna página para el dia anterior, simplemente no indiques la fecha:

.. code-block:: bash

   python train-report-smartbridges.py --bridge_path /ruta/al/puente --regenerar-hora 12:34:56 13:45:00

Salida
------

El informe PDF se guarda en:

::

   /ruta/al/puente/report/<año>/<mes>/<día>/train_report_<fecha>.pdf

Notas importantes
-----------------

**Estructura de los datos**
~~~~~~~~~~~~~~~~~~~~~~~~~~~
El script espera una estructura de carpetas específica para procesar los datos:

1. La carpeta raíz debe contener las carpetas que representan los puentes, cada una con dos subcarpetas:

   - La carpeta `raw` con los datos crudos de los sensores.
   - La carpeta `report` donde se guardan los informes generados.

2. La carpeta `raw` contiene subcarpetas organizadas por año, mes (inglés en minúscula) y día, y dentro de estas, las carpetas de los sensores.
3. Dentro de las carpetas de los sensores, existe la carpeta `anomalias` que contiene archivos CSV con datos no válidos o irrelevantes, los cuales son ignorados por el script.
4. La carpeta `report` contiene subcarpetas organizadas por año, mes (inglés en minúscula) y día.

**Formato de los archivos CSV**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Cada archivo CSV que contiene los datos en crudo se denomina `acceleration_HH-MM-SS.csv` y es un archivo único separado por comas (`,`), con números en formato decimal con punto (`.`). Las columnas son:
  
  - `timestamp`: Hora de la medición
  - `x_accel (g)`: Aceleración en el eje X (en g)
  - `y_accel (g)`: Aceleración en el eje Y (en g)
  - `z_accel (g)`: Aceleración en el eje Z (en g)

- El timestamp sigue el formato `HH:mm:ss.SSS` (hours:minutes:seconds.milliseconds)

**Procesamiento de eventos**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- El script procesa todos los archivos CSV de aceleraciones de todas las carpetas de sensores del día correspondiente.
- Se crean grupos disjuntos de vibraciones por timestamp (con una diferencia máxima de 2 segundos entre sensores) para determinar qué vibraciones corresponden al paso de un tren.
- Si no se encuentran suficientes sensores para un evento (por defecto, menos de 5), ese grupo se omite y no se genera una página en el informe.

**Funcionamiento y salida**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Se generan gráficas de aceleración y transformadas FFT para cada eje (X, Y, Z) de cada tren detectado.
- El PDF generado se guarda en la carpeta `report` del puente, organizada por año, mes y día. Cada página del informe corresponde a un tren detectado e incluye:
  - Número de tren, fecha y hora.
  - Gráficas de aceleración en los ejes X, Y y Z.
  - Transformada FFT para cada eje.
- Los archivos PDF generados se nombran con el formato `train_report_<fecha>.pdf`, donde `<fecha>` es la fecha del informe en formato `YYYYMMDD`.

**Cuestiones técnicas**
~~~~~~~~~~~~~~~~~~~~~~~~~~
- El script utiliza mutithreading para procesar cada carpeta de puente.
- El script está pensando para ser gestionado por un servicio de systemd que se active diariamente y automáticamente para generar informes diarios.
- El cálculo de la transformada FFT se realiza utilizando la función `scipy.fftpack.fft` para cada eje de aceleración que internamente utiliza multiprocessing.
- Se ha aplicado un multiprocessing extra para calcular la FFT por cada sensor.

