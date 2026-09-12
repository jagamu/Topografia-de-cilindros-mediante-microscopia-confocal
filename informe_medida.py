"""
informe_medida.py — informe de errores.

Este programa analiza los ficheros metricas_v4.18.txt y desarrollo_plano_v4.18.txt para obtener
un informe que nos ayuda a valorar si la fusión a sido correcta.

Una breve explicación de los resultados que vamos a obtener:
    1º. ¿Encajan entre si las capturas continuas? El RMS nos permite saber como de preciso ha sido el encaje
    2º. ¿Existe una deriva en la cadena de registro? Para conocer la deriva exial, realizamos la suma acumulada de todas las capturas
    3º. ¿Cierra el conjunto de muestras tras completar una vuelta? Podemos observarlo gracias al giro de cierre
    4º. ¿Tiene nuestro cilindro deformaciones? Para saberlo, analizamos los armonicos por vuelta
    5º. ¿Tiene el tamaño correcto? Comparamos el radio de nuestro software con el radio medido con el micrometro
"""

import glob
import os
import re
import sys
import tempfile
import time

import numpy as np

# Un giro par a par que supere este multiplo de la mediana se marca
UMBRAL_ATIPICO = 4.0
# Tolerancia al comparar el giro de cierre con un multiplo del paso
TOL_PASO = 0.25


# ════════════════════════════════════════════════════════════
# LECTURA
# ════════════════════════════════════════════════════════════

def buscar(carpeta, patron):
    hits = sorted(glob.glob(os.path.join(carpeta, patron)))
    return hits[-1] if hits else None


def leer_metricas(carpeta):
    """Extraemos el resumen y la tabla de solapes de metricas_v*.txt."""
    ruta = buscar(carpeta, "metricas_v*.txt")
    if ruta is None:
        raise SystemExit("No se encuentra archivo metricas_v*.txt en la carpeta.\n"
                         "Ejecuta antes el programa de reconstruccion.")
    texto = open(ruta, encoding="utf-8", errors="replace").read()

    def num(etiqueta):
        m = re.search(etiqueta + r"[^:]*:\s*\(?\s*(-?[\d.]+)", texto)
        return float(m.group(1)) if m else None

    resumen = {
        "fichero": os.path.basename(ruta),
        "n_muestras": num(r"N.{0,3} de muestras"),
        "radio_um": num(r"Radio ajustado"),
        "longitud_um": num(r"Longitud del cilindro"),
        "excentricidad_um": num(r"Excentricidad"),
        "desv_min_nm": num(r"Desv\. m.{0,3}n\."),
        "desv_max_nm": num(r"Desv\. m.{0,3}x\."),
    }


    num_re = r"[-+]?[\d.]+"
    patron = re.compile(
        r"Solape\s+(\S+)\s*:\s*RMS=\s*(" + num_re + r")"
        r".*?\(\s*(\d+)\s*correspondencias\s*\)"
        r".*?giro_eje=\s*(" + num_re + r")"
        r".*?[^n]=\(\s*(" + num_re + r")\s*,\s*(" + num_re + r")\s*,"
        r"\s*(" + num_re + r")\s*\)\s*.?m"
    )
    solapes = []
    for linea in texto.splitlines():
        m = patron.search(linea)
        if m:
            solapes.append({
                "par": m.group(1),
                "rms": float(m.group(2)),
                "n_corr": int(m.group(3)),
                "giro": float(m.group(4)),
                "dx": float(m.group(5)),
                "dy": float(m.group(6)),
                "dz": float(m.group(7)),
                "frio": "arranque en fr" in linea.lower(),
                "cierre": m.group(1).startswith("cierre"),
            })
    return resumen, solapes


def leer_desarrollo(carpeta, radio_um):
    """Lee el desarrollo plano si existe. Devuelve (angulos, desviaciones)."""
    ruta = buscar(carpeta, "desarrollo_plano_v*.txt")
    if ruta is None or radio_um is None:
        return None, None

    with open(ruta, encoding="utf-8", errors="replace") as f:
        for primera in f:
            if primera.strip():
                break
    limpia = primera.lstrip("#").strip()
    sep = next((c for c in (";", ",", "\t") if c in limpia), None)
    trozos = limpia.split(sep) if sep else limpia.split()
    try:
        [float(t.replace(",", ".")) for t in trozos]
        saltar = 1 if primera.lstrip().startswith("#") else 0
    except ValueError:
        saltar = 1

    print("  Leyendo el desarrollo plano...")
    try:
        import pandas as pd
        d = pd.read_csv(ruta, sep=(r"\s+" if sep is None else sep),
                        engine=("python" if sep is None else "c"),
                        header=None, skiprows=saltar, dtype=float).to_numpy()
    except Exception:
        d = np.loadtxt(ruta, delimiter=sep, skiprows=saltar)

    d = d[np.all(np.isfinite(d[:, :3]), axis=1)]
    ang = d[:, 1] / radio_um
    return np.arctan2(np.sin(ang), np.cos(ang)), d[:, 2]


# ════════════════════════════════════════════════════════════
# APARTADOS
# ════════════════════════════════════════════════════════════

def ap1_consistencia(pares, avisos):
    if not pares:
        return ("   Sin datos de ICP: el registro estaba desactivado.\n"
                "   Sin ICP no hay medida de consistencia entre capturas.\n")
    rms = np.array([p["rms"] for p in pares])
    corr = np.array([p["n_corr"] for p in pares])
    peor = pares[int(np.argmax(rms))]["par"]
    frios = [p["par"] for p in pares if p["frio"]]

    t = (f"   Pares evaluados                : {len(pares)}\n"
         f"   RMS mediano                    : {np.median(rms):.4f} um\n"
         f"   RMS minimo / maximo            : {rms.min():.4f} / {rms.max():.4f} um"
         f"   (peor: par {peor})\n"
         f"   Correspondencias por par       : {corr.min():,} a {corr.max():,}\n"
         f"   Arranques en frio              : {len(frios)}\n")
    if frios:
        t += f"     -> {', '.join(frios)}\n"
        avisos.append(f"{len(frios)} par(es) con arranque en frio: umbral "
                      "relajado, resultado no fiable.")
    t += ("OJO: un RMS bajo NO garantiza un registro correcto. \n")
    return t


def ap2_deriva(pares, resumen, avisos):
    if not pares:
        return "   No aplicable sin ICP.\n"
    dx = np.array([p["dx"] for p in pares])
    giros = np.array([p["giro"] for p in pares])
    n_neg = int((dx < 0).sum())
    n_pos = int((dx > 0).sum())
    sesgo = "negativo" if n_neg > n_pos else "positivo"
    dominante = max(n_neg, n_pos)

    t = (f"   Desplazamiento axial acumulado : {dx.sum():+.2f} um\n"
         f"   Signo de los desplazamientos   : {n_neg} negativos, "
         f"{n_pos} positivos\n"
         f"   Desplazamiento medio por par   : {dx.mean():+.4f} um\n"
         f"   Giro acumulado por la cadena   : {giros.sum():+.4f} grados\n")

    if dominante >= 0.9 * len(dx) and len(dx) >= 10:
        t += (f"   -> SESGO SISTEMATICO: {dominante} de {len(dx)} pares se\n"
              f"      desplazan en sentido {sesgo}. No es ruido.\n")
        avisos.append(f"Deriva axial sistematica: {dx.sum():+.1f} um "
                      f"acumulados en {len(dx)} pares.")

    if resumen.get("n_muestras") and resumen.get("longitud_um"):
        t += (f"   Longitud axial reconstruida    : "
              f"{resumen['longitud_um']:.2f} um\n"
              "      Comparar con la longitud del campo del objetivo: si es\n"
              "      mayor, el ICP ha estirado el cilindro deslizando capturas\n"
              "      a lo largo del eje, donde la superficie no lo restringe.\n")
    return t


def ap3_cierre(cierre, paso_deg, avisos):
    if cierre is None:
        return ("   No calculado: barrido parcial, o ICP desactivado.\n")
    g = cierre["giro"]
    d = np.linalg.norm([cierre["dx"], cierre["dy"], cierre["dz"]])
    t = (f"   Giro de cierre medido          : {g:+.4f} grados\n"
         f"   Traslacion de cierre           : {d:.2f} um\n"
         f"   RMS en el solape de cierre     : {cierre['rms']:.4f} um\n")
    if cierre["frio"]:
        t += "   Obtenido con ARRANQUE EN FRIO: no es fiable, revisar manualmente.\n"

    if paso_deg:
        k = abs(g) / paso_deg
        t += f"   Giro de cierre en pasos        : {k:.2f} pasos de {paso_deg:.3f} grados\n"
        if k > 0.5 and abs(k - round(k)) < TOL_PASO and round(k) >= 1:
            t += (f"   -> El cierre equivale a {round(k)} paso(s) completo(s).\n"
                  "      Sintoma de deslizamiento: el ICP ha encajado la\n"
                  "      ultima captura sobre una vecina equivocada.\n")
            avisos.append(f"Cierre deslizado {round(k)} paso(s) completo(s) "
                          f"({g:+.2f} grados). Reconstruccion no valida.")
        elif abs(g) > 2.0:
            avisos.append(f"Giro de cierre de {g:+.2f} grados, excesivo.")
    t += ("   El programa REPARTE esta correccion entre todas las capturas.\n"
          "   la reconstrucción puede arrastrar una torsion erronea.\n")
    return t


def ap4_atipicos(pares, avisos):
    if not pares:
        return "   No aplicable sin ICP.\n"
    fiables = [p for p in pares if not p["frio"]]
    med = np.median([abs(p["giro"]) for p in fiables]) if fiables else 0.0
    lim = max(UMBRAL_ATIPICO * med, 0.5)
    malos = sorted([p for p in pares if abs(p["giro"]) > lim],
                   key=lambda p: -abs(p["giro"]))

    t = (f"   Giro mediano de los pares      : {med:.4f} grados\n"
         f"   Umbral de sospecha             : {lim:.4f} grados\n"
         f"   Pares por encima del umbral    : {len(malos)} de {len(pares)}\n")
    if malos:
        t += "\n     par        giro (grados)   desplazamiento (um)   RMS\n"
        for p in malos[:10]:
            dd = np.linalg.norm([p["dx"], p["dy"], p["dz"]])
            t += (f"     {p['par']:<10s} {p['giro']:+10.4f}   "
                  f"{dd:12.2f}        {p['rms']:.3f}\n")
        t += ("\n   \n")

        avisos.append(f"{len(malos)} par(es) con giro atipico; revisar.")
    return t


def ap5_forma(ang, desv, n_arm=3, n_bins=720, pct=99.5):
    if ang is None:
        return ("   No se ha encontrado desarrollo_plano_v*.txt.\n"
                "   Sin el no puede analizarse la forma.\n"), None, None
    umbral = np.percentile(np.abs(desv), pct)
    keep = np.abs(desv) <= umbral
    idx = np.clip(((ang[keep] + np.pi) / (2*np.pi) * n_bins).astype(int),
                  0, n_bins - 1)
    suma = np.bincount(idx, weights=desv[keep], minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    ocup = cnt > 0
    perfil = np.full(n_bins, np.nan)
    perfil[ocup] = suma[ocup] / cnt[ocup]
    centros = -np.pi + (np.arange(n_bins) + 0.5) * (2*np.pi/n_bins)

    tt = centros[ocup]
    cols = [np.ones_like(tt)]
    for k in range(1, n_arm + 1):
        cols += [np.cos(k*tt), np.sin(k*tt)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), perfil[ocup], rcond=None)

    modelo = np.full_like(desv, coef[0])
    for k in range(1, n_arm + 1):
        modelo += coef[2*k-1]*np.cos(k*ang) + coef[2*k]*np.sin(k*ang)
    res = desv - modelo

    def est(v):
        v = v[np.abs(v) <= np.percentile(np.abs(v), pct)]
        return float(np.mean(np.abs(v))), float(np.sqrt(np.mean(v**2)))

    ra_a, _ = est(desv)
    ra_d, rq_d = est(res)
    nombres = {1: "excentricidad de montaje", 2: "ovalizacion de la pieza",
               3: "lobulado triangular"}
    t = ""
    for k in range(1, n_arm + 1):
        t += (f"   k={k} ({nombres.get(k, 'armonico ' + str(k)):<24s}): "
              f"{np.hypot(coef[2*k-1], coef[2*k]):8.4f} um\n")
    t += (f"   Desviacion media ANTES         : {ra_a:.4f} um\n"
          f"   Desviacion media DESPUES       : {ra_d:.4f} um\n"
          f"   Valor cuadratico medio DESPUES : {rq_d:.4f} um\n"
          f"   Puntos descartados como picos  : {100*(1-keep.mean()):.2f} %\n"
          "   Solo la cifra DESPUES puede presentarse como rugosidad, y aun\n"
          "   asi contiene el error sistematico del propio microscopio.\n")
    return t, np.rad2deg(centros), perfil


def ap6_exactitud(radio, ref, avisos):
    if ref is None:
        return ("   Sin referencia externa.\n"
                "   Los apartados 1 a 4 miden CONSISTENCIA, no exactitud: un\n")
    err = radio - ref
    t = (f"   Radio reconstruido             : {radio:.4f} um\n"
         f"   Radio de referencia            : {ref:.4f} um\n"
         f"   Error                          : {err:+.4f} um "
         f"({100*err/ref:+.2f} %)\n")
    if abs(err) < 0.5:
        t += ("\n")
    elif abs(100*err/ref) > 1.0:
        avisos.append(f"Radio desviado {100*err/ref:+.2f} % respecto del patron.")
    return t


# ════════════════════════════════════════════════════════════

def escribible(carpeta, nombre):
    raiz, ext = os.path.splitext(nombre)
    for ruta in (os.path.join(carpeta, nombre),
                 os.path.join(carpeta, f"{raiz}_{time.strftime('%H%M%S')}{ext}"),
                 os.path.join(tempfile.gettempdir(), nombre)):
        try:
            with open(ruta, "w", encoding="utf-8"):
                pass
            return ruta
        except OSError:
            continue
    return None


def pedir_angulo_vuelta(defecto=360.0):

    txt = input(f"  Angulo total de la vuelta en grados "
                f"(Intro para {defecto:g}): ").strip()
    if not txt:
        return defecto
    try:
        valor = float(txt.replace(",", "."))
    except ValueError:
        print(f"  Valor no reconocido, se usa {defecto:g} grados.")
        return defecto
    if valor <= 0:
        print(f"  El angulo debe ser positivo; se usa {defecto:g} grados.")
        return defecto
    return valor


def main(carpeta):
    print(f"\nCarpeta: {carpeta}")
    resumen, solapes = leer_metricas(carpeta)
    print(f"  Leido {resumen['fichero']}: {len(solapes)} solapes.")

    ref = input("  Radio de referencia externa en um "
                "(micrometro; Intro si no procede): ").strip()
    ref = float(ref.replace(",", ".")) if ref else None

    angulo_vuelta = pedir_angulo_vuelta()

    pares = [s for s in solapes if not s["cierre"]]
    cierre = next((s for s in solapes if s["cierre"]), None)
    paso = (angulo_vuelta / resumen["n_muestras"]
            if resumen.get("n_muestras") else None)

    ang, desv = leer_desarrollo(carpeta, resumen.get("radio_um"))

    avisos = []
    t1 = ap1_consistencia(pares, avisos)
    t2 = ap2_deriva(pares, resumen, avisos)
    t3 = ap3_cierre(cierre, paso, avisos)
    t4 = ap4_atipicos(pares, avisos)
    t5, perfil_x, perfil_y = ap5_forma(ang, desv)
    t6 = ap6_exactitud(resumen["radio_um"], ref, avisos)

    cab = (f"   Capturas                       : "
           f"{int(resumen['n_muestras']) if resumen.get('n_muestras') else '?'}\n"
           f"   Paso angular deducido          : "
           f"{paso:.4f} grados ({angulo_vuelta:g} / n)\n" if paso else "")
    cab += (f"   Radio ajustado                 : {resumen['radio_um']:.4f} um\n"
            f"   Longitud axial                 : {resumen['longitud_um']:.2f} um\n")

    inf = ("=" * 66 + "\nINFORME DE ERRORES DE LA MEDIDA\n" + "=" * 66 + "\n\n"
           "DATOS DE LA RECONSTRUCCION\n" + cab + "\n"
           "1. CONSISTENCIA ENTRE CAPTURAS VECINAS \n" + t1 + "\n"
           "2. DERIVA DE LA CADENA DE REGISTRO \n" + t2 + "\n"
           "3. CIERRE DEL BUCLE \n" + t3 + "\n"
           "4. PARES DE REGISTRO ATIPICOS \n" + t4 + "\n"
           "5. DESVIACION RESPECTO DEL CILINDRO IDEAL \n" + t5 + "\n"
           "6. EXACTITUD \n" + t6 + "\n"
           + "=" * 66 + "\nVALORACION\n" + "=" * 66 + "\n")
    if avisos:
        inf += "   Esta medida presenta los siguientes problemas:\n\n"
        for i, a in enumerate(avisos, 1):
            inf += f"   {i}. {a}\n"
        inf += ("\n   Revisa los apartados correspondientes antes de dar por\n"
                "   buena la reconstruccion.\n")
    else:
        inf += "   No se han detectado anomalias en las comprobaciones automaticas.\n"
    inf += "=" * 66 + "\n"

    print("\n" + inf)

    ruta = escribible(carpeta, "informe_medida.txt")
    if ruta:
        open(ruta, "w", encoding="utf-8").write(inf)
        print(f"Informe guardado en : {ruta}")
    else:
        print("AVISO: no se ha podido guardar. El texto de arriba es el informe.")

    if perfil_x is not None:
        rp = escribible(carpeta, "perfil_angular.csv")
        if rp:
            with open(rp, "w", encoding="utf-8") as f:
                f.write("angulo_deg;desviacion_media_um\n")
                for a, v in zip(perfil_x, perfil_y):
                    f.write(f"{a:.4f};" + ("" if not np.isfinite(v)
                                           else f"{v:.6f}") + "\n")
            print(f"Perfil angular en   : {rp}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else input("Carpeta de trabajo: ").strip())