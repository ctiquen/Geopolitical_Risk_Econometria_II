# Trabajo 1 - Econometría II (2026-2S)

Modelamiento ARMA del Índice de Riesgo Geopolítico (GPR) con la metodología Box-Jenkins:
identificación, estimación, validación y pronóstico a 10 meses.

Lyna Camila Tique Nova

## Datos

- Serie: Índice de Riesgo Geopolítico global (`GPR`), mensual, enero de 1985 a septiembre de 2026 (501 observaciones).
- Fuente: Caldara y Iacoviello, *Geopolitical Risk Index* (https://www.matteoiacoviello.com/gpr.htm).
- Archivo incluido en este proyecto: `data/data_gpr_export.xls`.

## Estructura del proyecto

```
.
├── analisis_gpr.py      # script único: pasos 1 a 5
├── requirements.txt     # paquetes necesarios
├── README.md
├── data/
│   └── data_gpr_export.xls
└── outputs/             # se crea solo al correr el script (gráficos y tablas)
```

## Cómo ejecutarlo

1. Abrir una terminal (Anaconda Prompt en Windows) y ubicarse en la carpeta del proyecto.
2. Instalar los paquetes: `python -m pip install -r requirements.txt`
3. Correr el script: `python analisis_gpr.py`

Las rutas son relativas, así que no hay que cambiar nada según el computador.
Al correr, se abren ventanas con gráficos: hay que cerrarlas para que el script termine.
Los gráficos y tablas quedan en la carpeta `outputs/`.

## Qué hace el script

1. Carga la serie y la grafica.
2. Identificación: pruebas ADF y KPSS, ACF y PACF.
3. Estimación: AR(1), ARMA(1,1) y AR(3), comparados con AIC y BIC.
4. Validación: Ljung-Box, Jarque-Bera, ARCH, gráficos de diagnóstico y prueba fuera de muestra.
5. Pronóstico a 10 meses con el modelo elegido (AR(3)).

## Referencia

Caldara, D. y Iacoviello, M. (2022). Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194-1225.
