# analisis_gpr.py  |  Pasos 1 a 5: cargar, identificar, estimar, validar y pronosticar la serie GPR
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")  # oculta avisos de versión que no afectan los resultados

# ---------------------------------------------------------------
# PASO 1. Cargar la serie y graficarla
# ---------------------------------------------------------------
# Rutas relativas: todo se mide desde la carpeta donde está este script
BASE = Path(__file__).resolve().parent
ruta_datos = BASE / "data" / "data_gpr_export.xls"
ruta_salidas = BASE / "outputs"
ruta_salidas.mkdir(exist_ok=True)

df = pd.read_excel(ruta_datos, engine="xlrd")
df["month"] = pd.to_datetime(df["month"])
y = df.set_index("month")["GPR"].dropna().asfreq("MS")

print(y.head())
print("Observaciones:", len(y), "| desde", y.index.min().date(), "hasta", y.index.max().date())

plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title("Índice de Riesgo Geopolítico (GPR) mensual")
plt.xlabel("Año")
plt.ylabel("GPR")
plt.tight_layout()
plt.savefig(ruta_salidas / "01_serie_gpr.png", dpi=200)

# ---------------------------------------------------------------
# PASO 2. Identificación: ¿es estacionaria? ¿qué dicen ACF y PACF?
# ---------------------------------------------------------------
def pruebas(serie, nombre):
    """ADF: H0 = raíz unitaria (NO estacionaria). KPSS: H0 = estacionaria."""
    p_adf = adfuller(serie, autolag="AIC")[1]
    p_kpss = kpss(serie, regression="c", nlags="auto")[1]
    print(f"{nombre:<20} ADF p = {p_adf:.4f} | KPSS p = {p_kpss:.4f}")

print("\n--- Pruebas de estacionariedad ---")
pruebas(y, "GPR en niveles")
pruebas(np.log(y), "log(GPR)")
pruebas(y.diff().dropna(), "Primera diferencia")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y, lags=36, ax=ax[0], title="ACF del GPR")
plot_pacf(y, lags=36, ax=ax[1], method="ywm", title="PACF del GPR")
plt.tight_layout()
plt.savefig(ruta_salidas / "02_acf_pacf.png", dpi=200)

# ---------------------------------------------------------------
# PASO 3. Estimación: ajustar los modelos candidatos y compararlos
# ---------------------------------------------------------------
# order = (p, d, q). d = 0 porque la serie ya es estacionaria en niveles.
candidatos = {"AR(1)": (1, 0, 0), "ARMA(1,1)": (1, 0, 1), "AR(3)": (3, 0, 0)}

resultados = {}
filas = []
for nombre, orden in candidatos.items():
    res = SARIMAX(y, order=orden, trend="c").fit(disp=False)
    resultados[nombre] = res
    no_signif = [k for k, v in res.pvalues.items() if v > 0.05 and k != "sigma2"]
    raices = list(np.abs(res.arroots)) + list(np.abs(res.maroots))
    filas.append({
        "Modelo": nombre,
        "AIC": round(res.aic, 1),
        "BIC": round(res.bic, 1),
        "Coef. no significativos": ", ".join(no_signif) or "ninguno",
        "Raíces fuera del círculo": "sí" if all(r > 1 for r in raices) else "no",
    })

tabla = pd.DataFrame(filas).set_index("Modelo")
print("\n--- Comparación de modelos candidatos ---")
print(tabla.to_string())
tabla.to_csv(ruta_salidas / "03_comparacion_modelos.csv", encoding="utf-8-sig")

modelo_ver = "AR(3)"  # cambia este nombre para ver el detalle de otro modelo
print(f"\n--- Detalle de {modelo_ver} ---")
print(resultados[modelo_ver].summary())

# ---------------------------------------------------------------
# PASO 4. Validación: ¿los residuales parecen ruido blanco?
# ---------------------------------------------------------------
# 4a. Pruebas sobre los residuales (se reportan p-valores)
#   Ljung-Box: H0 = no hay autocorrelación (queremos p > 0.05)
#   Jarque-Bera: H0 = residuales normales
#   ARCH: H0 = varianza constante (sin efectos ARCH)
filas_v = []
for nombre, orden in candidatos.items():
    e = resultados[nombre].resid
    gl = orden[0] + orden[2]  # parámetros ARMA estimados (p + q)
    lb = acorr_ljungbox(e, lags=[10, 20, 24], model_df=gl)["lb_pvalue"]
    e_sin_extremo = e.drop(e.abs().idxmax())  # quita el residual más grande
    filas_v.append({
        "Modelo": nombre,
        "LB(10)": lb.iloc[0], "LB(20)": lb.iloc[1], "LB(24)": lb.iloc[2],
        "Jarque-Bera": jarque_bera(e)[1],
        "ARCH(12)": het_arch(e, nlags=12)[1],
        "ARCH(12) sin extremo": het_arch(e_sin_extremo, nlags=12)[1],
    })

tabla_v = pd.DataFrame(filas_v).set_index("Modelo").round(3)
print("\n--- Validación de residuales (p-valores) ---")
print(tabla_v.to_string())
tabla_v.to_csv(ruta_salidas / "04_validacion_residuales.csv", encoding="utf-8-sig")

# 4b. Gráficos de diagnóstico del modelo elegido
modelo_final = "AR(3)"  # el que elegimos mirando la tabla de arriba
resultados[modelo_final].plot_diagnostics(figsize=(12, 8), lags=24)
plt.tight_layout()
plt.savefig(ruta_salidas / "04_diagnostico_residuales.png", dpi=200)

# 4c. Prueba fuera de muestra: dejamos los últimos 12 meses sin usar,
#     pronosticamos esos 12 meses y comparamos con lo que realmente pasó
h = 12
entrenamiento, prueba = y.iloc[:-h], y.iloc[-h:]

def errores(pred):
    return np.sqrt(((prueba - pred) ** 2).mean()), (prueba - pred).abs().mean()

filas_h = []
for nombre, orden in candidatos.items():
    pred = SARIMAX(entrenamiento, order=orden, trend="c").fit(disp=False).forecast(h)
    filas_h.append((nombre, *errores(pred)))
filas_h.append(("Referencia: media", *errores(pd.Series(entrenamiento.mean(), index=prueba.index))))
filas_h.append(("Referencia: último valor", *errores(pd.Series(entrenamiento.iloc[-1], index=prueba.index))))

tabla_h = pd.DataFrame(filas_h, columns=["Modelo", "RMSE", "MAE"]).set_index("Modelo").round(1)
print(f"\n--- Error fuera de muestra ({prueba.index[0].date()} a {prueba.index[-1].date()}) ---")
print(tabla_h.to_string())
tabla_h.to_csv(ruta_salidas / "04_error_fuera_de_muestra.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------
# PASO 5. Pronóstico a 10 períodos con el modelo elegido
# ---------------------------------------------------------------
# El modelo final se estima con TODA la muestra (ya no dejamos meses por fuera)
pron = resultados[modelo_final].get_forecast(steps=10)
pred = pron.predicted_mean
ic = pron.conf_int(alpha=0.05)  # intervalo de confianza del 95 %

tabla_p = pd.DataFrame({
    "Pronóstico": pred,
    "IC 95% inferior": ic.iloc[:, 0],
    "IC 95% superior": ic.iloc[:, 1],
}).round(1)
tabla_p.index = tabla_p.index.strftime("%Y-%m")
tabla_p.index.name = "Mes"
print(f"\n--- Pronóstico a 10 meses con {modelo_final} ---")
print(tabla_p.to_string())
tabla_p.to_csv(ruta_salidas / "05_pronostico_10_meses.csv", encoding="utf-8-sig")

# Gráfico: últimos 5 años + pronóstico con su banda
reciente = y.iloc[-60:]
plt.figure(figsize=(10, 4.5))
plt.plot(reciente, label="GPR observado")
plt.plot(pred, color="tab:red", label=f"Pronóstico {modelo_final}")
plt.fill_between(pred.index, ic.iloc[:, 0], ic.iloc[:, 1],
                 color="tab:red", alpha=0.2, label="IC 95 %")
plt.axhline(y.mean(), color="gray", linestyle="--", linewidth=1, label="Media histórica")
plt.title("Pronóstico del GPR a 10 meses")
plt.xlabel("Fecha")
plt.ylabel("GPR")
plt.legend()
plt.tight_layout()
plt.savefig(ruta_salidas / "05_pronostico.png", dpi=200)

plt.show()  # abre las ventanas con los gráficos; ciérralas para terminar