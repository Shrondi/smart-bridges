Estructura del repositorio
==========================

A continuación se muestra la estructura principal del directorio del proyecto ``smart-bridges`` y una breve descripción de cada carpeta y archivo relevante:

.. code-block:: text

   smart-bridges/
   ├── docs/
   │   └── ...                # Documentación generada y archivos fuente Sphinx
   ├── train-report/
   │   ├── train-report-smartbridges.py   # Script principal para generación de informes PDF
   │   ├── EXPLANATION.md     # Explicación detallada del funcionamiento del script
   │   └── README.md          # Información y guía de uso del módulo train-report
   ├── .gitignore             # Exclusiones de archivos para el control de versiones
   ├── LICENSE                # Licencia del proyecto
   ├── README.md              # Descripción general y guía rápida del repositorio
   └── requirements.txt       # Dependencias necesarias para el proyecto

.. rubric:: Descripción de carpetas y archivos

- **docs/**  
  Contiene la documentación del proyecto, incluyendo la configuración y los archivos fuente para Sphinx.

- **train-report/**  
  Módulo principal para el procesamiento de datos y generación de informes PDF a partir de los datos de acelerómetros.

  - ``train-report-smartbridges.py``  
    Script principal para la generación de informes de eventos de tren detectados por los sensores.

  - ``requirements.txt``  
    Archivo con las dependencias específicas necesarias para el módulo ``train-report``.

  - ``README.md``  
    Guía de uso y descripción específica del módulo ``train-report``.

- **.gitignore**  
  Lista de archivos y carpetas que serán ignorados por el sistema de control de versiones Git.

- **LICENSE**  
  Archivo con la licencia bajo la que se distribuye el proyecto.

- **README.md**  
  Descripción general, objetivos y guía rápida de uso del repositorio.
