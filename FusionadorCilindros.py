import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # cambia a 'Qt5Agg' si prefieres Qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib import cm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
import os, glob, re, threading

# ── En caso de no tener Open3D instalado se usa mathplotlib ──────────
try:
    import open3d as o3d
    _O3D_DISPONIBLE = True
except ImportError:
    _O3D_DISPONIBLE = False
    print("[AVISO] open3d no encontrado. Instálalo con:  pip install open3d")
    print("        La vista 3D usará matplotlib como fallback.\n")
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

# ════════════════════════════════════════════════════════════
# PARÁMETROS CONFIGURABLES
# ════════════════════════════════════════════════════════════

DIRECTORIO_DATOS          = ".DatosParaAnalizar/10x eje 60% solape 2.45/txt"
PATRON_ARCHIVO            = "TopoInd_1_*.txt"
ANGULO_TOTAL_GRADOS       = 360.0
SOLAPAMIENTO              = 0.599545        # 10–20 % (hasta 60 % soportado)

# Sentido de giro del solape de las distintas muestras. Configurable editando este valor directamente:
# SENTIDO_GIRO = -1  →  giro HORARIO       (visto desde X+ hacia X-)
# SENTIDO_GIRO = +1  →  giro ANTIHORARIO   (visto desde X+ hacia X-)
SENTIDO_GIRO: int         = -1

ESTIMAR_CENTRO            = True
CENTRO_MANUAL             = (0.0, 0.0)

# En caso de conocer el radio en µm de la muestra dejar esta configuración en none
RADIO_CONOCIDO_UM         = None

#Estimación del radio haciendo uso de la primera captura
USAR_SOLO_PRIMERA_CAPTURA = True

USAR_ICP                  = True
ICP_MAX_ITER              = 50
ICP_TOLERANCIA            = 1e-6
ICP_DIST_MAX_UM           = 0.5

# Factor máximo de relajación del umbral de correspondencia para el
# "arranque en frío" del ICP: si con ICP_DIST_MAX_UM no hay ni
# siquiera 10 correspondencias en la posición de partida, se aumenta progresivamente
# umbrales más laxos (×2, ×4, ×8...). Esto se hace debido a la inclinación de las muestras

ICP_FACTOR_MAX_RELAJACION = 15.0

# Ángulo máximo (grados) de rotación de la captura concreta. Este valor impide que nuestra
# captura gire 180 grados
GIRO_MAX_GRADOS           = 45.0

# distancia (µm) para fusionar/promediar puntos duplicados
DISTANCIA_DUPLICADOS_UM   = 0.05

# Esta constante nos ayuda a corregir la desviación de las muestras con respecto al cilindrio ideal inicial
REAJUSTAR_CENTRO_POST_FUSION = True

N_SECCIONES_ANGULARES     = 72
N_SECCIONES_AXIALES       = 80

# ── Configuraciones del color map ───────────────────────────────
# jet            : azul (debajo) → cian → verde (ideal) → amarillo → rojo (encima)
# turbo         : como jet pero perceptualmente más uniforme;
#                   extremo bajo violeta (no azul puro)
# rainbow       : extremo bajo entre azul y magenta (no azul puro)
# gist_rainbow  : orden de color distinto, extremo bajo rojo (NO válido
#                   para este esquema sin invertir con '_r')

COLORMAP                  = 'jet'

# Percentil simétrico para la escala de color
PERCENTIL_COLOR           = 98

# ── Parámetros de malla Open3D ────────────────────────────────────────────────
# Método de reconstrucción: 'poisson' (más denso, sin agujeros) o 'bpa' (rápido)
METODO_MESH               = 'poisson'

# Poisson: profundidad del octree. 8→rápido/menos detalle, 9→equilibrio, 10→fino
POISSON_DEPTH             = 9

# Poisson: fracción de vértices de baja densidad, elimina bordes curvados
POISSON_TRIM              = 0.02

# BPA: factores sobre la distancia media al vecino más próximo para los radios
BPA_RADIOS_FACTOR         = [1.0, 2.0, 4.0]

# ── Parámetros específicos de Open3D ─────────────────────────────────────────
O3D_POINT_SIZE            = 1.5
O3D_BG_COLOR              = [0.04, 0.05, 0.07]
O3D_VENTANA_ANCHO         = 1100
O3D_VENTANA_ALTO          = 800
O3D_N_ANILLO              = 300
O3D_COLOR_ANILLO          = [1.0, 0.85, 0.0]   # amarillo

# Factor de exageración vertical del relieve en la vista de desarrollo plano
EXAGERACION_VERTICAL_DESARROLLO = 7.5

# ════════════════════════════════════════════════════════════
# UTILIDADES DE CARGA Y LIMPIEZA
# ════════════════════════════════════════════════════════════

def limpiar_nube(pts: np.ndarray) -> np.ndarray:
    mask = np.all(np.abs(pts) < 1e6, axis=1) & np.all(np.isfinite(pts), axis=1)
    return pts[mask]

def cargar_archivo(ruta: str) -> np.ndarray:
    d = np.loadtxt(ruta)
    return limpiar_nube(d.reshape(-1, 3) if d.ndim == 1 else d)

def cargar_capturas(directorio: str, patron: str) -> list[tuple[int, np.ndarray]]:
    archivos = glob.glob(os.path.join(directorio, patron))
    if not archivos:
        raise FileNotFoundError(f"Sin archivos: {os.path.join(directorio, patron)}")
    re_num = re.compile(r"TopoInd_\d+_(\d+)\.txt")
    capturas = []
    for a in archivos:
        m = re_num.search(os.path.basename(a))
        if m:
            capturas.append((int(m.group(1)), cargar_archivo(a)))
    capturas.sort(key=lambda x: x[0])
    print(f"[OK] {len(capturas)} capturas: {[c[0] for c in capturas]}")
    return capturas

# ════════════════════════════════════════════════════════════
# GEOMETRÍA DEL CILINDRO
# ════════════════════════════════════════════════════════════

def ajustar_circulo(
    yz: np.ndarray, R_fijo: float | None = None
) -> tuple[float, float, float]:
    y, z = yz[:, 0], yz[:, 1]
    if R_fijo is None:
        def res(p): return np.sqrt((y-p[0])**2 + (z-p[1])**2) - p[2]
        r = least_squares(res, [y.mean(), z.mean(), np.std(y)], loss='soft_l1')
        return r.x[0], r.x[1], abs(r.x[2])
    else:
        def res(p): return np.sqrt((y-p[0])**2 + (z-p[1])**2) - R_fijo
        r = least_squares(res, [y.mean(), z.mean()], loss='soft_l1')
        return r.x[0], r.x[1], R_fijo

def estimar_centro(
    capturas: list[tuple[int, np.ndarray]], R_fijo: float | None, primera: bool
) -> tuple[tuple[float, float], float]:
    yz = capturas[0][1][:, [1,2]] if primera else np.vstack([p[:,[1,2]] for _,p in capturas])
    yc, zc, R = ajustar_circulo(yz, R_fijo)
    print(f"  Centro=({yc:.4f},{zc:.4f}) µm  R={R:.4f} µm")
    return (yc, zc), R

def calcular_angulos(n: int, total_deg: float, sentido: int = -1) -> np.ndarray:
    if sentido not in (1, -1):
        raise ValueError(
            f"sentido debe ser -1 (horario) o 1 (antihorario), recibido: {sentido}"
        )
    if n < 1:
        raise ValueError(f"n debe ser >= 1, recibido: {n}")
    if n == 1:
        return np.array([0.0])
    paso = total_deg / n
    return np.deg2rad([sentido * i * paso for i in range(n)])

def rotar_X(
    pts: np.ndarray, ang: float, centro: tuple[float, float]
) -> np.ndarray:
    yc, zc = centro
    y, z = pts[:,1]-yc, pts[:,2]-zc
    c, s = np.cos(ang), np.sin(ang)
    return np.column_stack((pts[:,0], y*c - z*s + yc, y*s + z*c + zc))

# ════════════════════════════════════════════════════════════
# ICP 2D + CORRECCIÓN DE CIERRE
# ════════════════════════════════════════════════════════════

def _dif_angular(a: np.ndarray, b: float) -> np.ndarray:

    return np.arctan2(np.sin(a - b), np.cos(a - b))

def _media_angular(angs: np.ndarray) -> float:

    return np.arctan2(np.mean(np.sin(angs)), np.mean(np.cos(angs)))

def _ventana_solape(paso_deg: float, solap: float) -> float:

    if not (0.0 <= solap < 1.0):
        raise ValueError(f"solap debe estar en [0,1), recibido: {solap}")
    if paso_deg <= 0:
        raise ValueError(f"paso_deg debe ser > 0, recibido: {paso_deg}")
    vent_deg = (paso_deg * solap)
    return np.deg2rad(vent_deg)

def _paso_kabsch(src_pts: np.ndarray, tgt_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    cs = src_pts.mean(axis=0)
    ct = tgt_pts.mean(axis=0)
    A = (src_pts - cs).T @ (tgt_pts - ct)
    U, _, Vt = np.linalg.svd(A)
    # Corrección de reflexión: sin esto, el SVD podría devolver una
    # reflexión en vez de una rotación pura si las correspondencias son
    # casi coplanares. Parte del algoritmo correcto, no una precaución de más.
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    sign = sign if sign != 0 else 1.0
    D = np.diag([1.0, 1.0, sign])
    R_step = Vt.T @ D @ U.T
    t_step = ct - R_step @ cs
    return R_step, t_step

def _icp_iterar(
    src_cur: np.ndarray,
    tgt: np.ndarray,
    tree: cKDTree,
    dist_umbral: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    R_acc = np.eye(3)
    t_acc = np.zeros(3)
    n_ultima = 0
    for _ in range(max_iter):
        d, idx = tree.query(src_cur, k=1)
        mask = d < dist_umbral
        n_ultima = int(mask.sum())
        if n_ultima < 10:
            break
        R_step, t_step = _paso_kabsch(src_cur[mask], tgt[idx[mask]])
        src_cur = (R_step @ src_cur.T).T + t_step
        R_acc = R_step @ R_acc
        t_acc = R_step @ t_acc + t_step
        ang_step = np.arccos(np.clip((np.trace(R_step) - 1.0) / 2.0, -1.0, 1.0))
        if ang_step < tol and np.linalg.norm(t_step) < tol:
            break
    return src_cur, R_acc, t_acc, n_ultima

def icp_3d(
    src: np.ndarray,
    tgt: np.ndarray,
    max_iter: int,
    tol: float,
    dist_max: float,
    factor_max_relajacion: float = 20.0,
    giro_max_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:

    if src.shape[1] != 3 or tgt.shape[1] != 3:
        raise ValueError("icp_3d espera nubes de puntos con 3 columnas (X,Y,Z)")

    tree = cKDTree(tgt)

    def _giro_total_deg(R: np.ndarray) -> float:
        return float(np.rad2deg(
            np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        ))

    def _metrica_final(src_final: np.ndarray) -> tuple[float, int]:
        d_final, _ = tree.query(src_final, k=1)
        mask_final = d_final < dist_max
        n_corr = int(mask_final.sum())
        rms = float(np.sqrt(np.mean(d_final[mask_final] ** 2))) if n_corr > 0 else float('nan')
        return rms, n_corr

    d0, idx0 = tree.query(src, k=1)
    n_inicial_estricto = int((d0 < dist_max).sum())

    if n_inicial_estricto >= 10:
        # ── Camino normal: no hace falta arranque en frío ──────────────────
        src_cur, R_total, t_total, _ = _icp_iterar(src.copy(), tgt, tree, dist_max, max_iter, tol)
        giro_total_deg = _giro_total_deg(R_total)
        if giro_total_deg > giro_max_deg:
            print(f"  [AVISO ICP] Resultado descartado por implausible: ángulo de "
                  f"rotación total {giro_total_deg:.2f}° > GIRO_MAX_GRADOS "
                  f"({giro_max_deg:.1f}°), sin haber necesitado arranque en frío. "
                  f"Esta pareja de capturas queda SIN CORREGIR -- caso inusual, "
                  f"revisar manualmente.")
            return np.eye(3), np.zeros(3), float('nan'), 0, False
        rms, n_corr = _metrica_final(src_cur)
        return R_total, t_total, rms, n_corr, False

    # ── Arranque en frío, PRIMER factor válido y plausible ──────────
    # Si queremos más pruebas, aumentar factor y factores_probados
    factores_probados = 0
    factor = 2
    while factor <= int(factor_max_relajacion):
        dist_relajado = dist_max * factor
        n_ini = int((d0 < dist_relajado).sum())
        if n_ini >= 10:
            factores_probados += 1
            src_c, R_fase1, t_fase1, _ = _icp_iterar(src.copy(), tgt, tree, dist_relajado, max_iter, tol)
            src_c, R_fase2, t_fase2, _ = _icp_iterar(src_c, tgt, tree, dist_max, max_iter, tol)
            R_cand = R_fase2 @ R_fase1
            t_cand = R_fase2 @ t_fase1 + t_fase2

            giro_cand = _giro_total_deg(R_cand)
            if giro_cand <= giro_max_deg:
                rms_cand, n_corr_cand = _metrica_final(src_c)
                if n_corr_cand > 0:
                    print(f"  [AVISO ICP] Arranque en frío: primer factor válido y "
                          f"plausible ×{factor} (de {factores_probados} probado(s)), "
                          f"RMS={rms_cand:.4f} µm. Desalineación inicial mayor que "
                          f"ICP_DIST_MAX_UM.")
                    return R_cand, t_cand, rms_cand, n_corr_cand, True
        factor += 1

    print(f"  [AVISO ICP] Ningún factor de relajación entre ×2 y "
          f"×{int(factor_max_relajacion)} dio un resultado con "
          f"correspondencias suficientes Y plausible (ángulo ≤ "
          f"{giro_max_deg:.1f}°) -- esta pareja de capturas queda SIN "
          f"CORREGIR. Revisar si hay algo anómalo en esa captura o en su "
          f"registro angular.")
    return np.eye(3), np.zeros(3), float('nan'), 0, False


def _aviso_consistencia_vecinas(informe: list[dict]) -> None:

    confianza = [abs(i['giro_eje_deg']) for i in informe if not i.get('arranque_frio', False)]
    if not confianza:
        return  # sin referencia fiable con la que comparar (caso raro)

    mediana_confianza = float(np.median(confianza))
    # Margen generoso (no es un criterio exacto): al menos 3x la mediana
    # de lo ya validado, con un suelo de 10° para no ser hipersensible
    # cuando las correcciones de confianza ya son casi cero.
    umbral = max(3.0 * mediana_confianza, 10.0)

    for info in informe:
        if not info.get('arranque_frio', False):
            continue
        giro_abs = abs(info['giro_eje_deg'])
        if giro_abs > umbral:
            print(f"  [AVISO ICP] Consistencia: el solape {info['par']} necesitó "
                  f"arranque en frío y su giro ({giro_abs:.2f}°) se desvía mucho "
                  f"del patrón de las parejas de confianza (mediana "
                  f"{mediana_confianza:.2f}°) -- revisar manualmente si la unión "
                  f"es físicamente correcta.")
            info['aviso_consistencia'] = (
                f"Giro {giro_abs:.2f}° muy por encima de la mediana de las "
                f"parejas de confianza ({mediana_confianza:.2f}°) -- revisar."
            )

def registrar_icp(
    nubes: list[np.ndarray],
    centro: tuple[float, float],
    angulos_nominales: np.ndarray,
    paso_deg: float,
    solap: float,
    max_iter: int,
    tol: float,
    dist_max: float,
    angulo_total_deg: float,
    factor_max_relajacion: float = 20.0,
    giro_max_deg: float = 45.0,
) -> tuple[list[np.ndarray], list[dict]]:

    print("\n[ICP] Registro entre capturas (3D, 6 GDL: rotación + traslación completas)...")
    yc, zc = centro
    n = len(nubes)
    vent = _ventana_solape(paso_deg, solap)
    print(f"  Ventana de solape: ±{np.rad2deg(vent):.3f}° "
          f"(paso={paso_deg:.4f}°, solape={solap*100:.3f}%)")

    resto = angulo_total_deg % 360.0
    bucle_cerrado = np.isclose(resto, 0.0, atol=1e-6) or np.isclose(resto, 360.0, atol=1e-6)

    def centrar(nube: np.ndarray) -> np.ndarray:
        # Solo Y,Z se centran en el eje del cilindro; X (axial) es directo.
        return np.column_stack([nube[:, 0], nube[:, 1] - yc, nube[:, 2] - zc])

    def euler_diagnostico(R: np.ndarray) -> np.ndarray:
        # Descomposición aproximada solo para diagnóstico por consola/
        # exportación (magnitud del giro vs. la inclinación); no pretende
        # ser una descomposición física rigurosa de la rotación.
        return Rotation.from_matrix(R).as_euler('xyz', degrees=True)

    R_chain: list[np.ndarray] = [np.eye(3)]
    t_chain: list[np.ndarray] = [np.zeros(3)]
    informe: list[dict] = []

    for i in range(1, n):
        src_full = centrar(nubes[i])
        tgt_full = centrar(nubes[i - 1])
        ang_src = np.arctan2(src_full[:, 2], src_full[:, 1])
        ang_tgt = np.arctan2(tgt_full[:, 2], tgt_full[:, 1])

        borde = _media_angular(
            np.array([angulos_nominales[i - 1], angulos_nominales[i]])
        )
        ms = np.abs(_dif_angular(ang_src, borde)) < vent
        mt = np.abs(_dif_angular(ang_tgt, borde)) < vent
        if ms.sum() < 20 or mt.sum() < 20:
            ms = mt = slice(None)

        R_pair, t_pair, rms, n_corr, arranque_frio = icp_3d(
            src_full[ms], tgt_full[mt], max_iter, tol, dist_max,
            factor_max_relajacion, giro_max_deg,
        )
        euler = euler_diagnostico(R_pair)
        print(f"  Par {i-1}→{i}: giro_eje={euler[0]:+.4f}°  "
              f"inclinación=({euler[1]:+.4f}°,{euler[2]:+.4f}°)  "
              f"Δ=({t_pair[0]:+.4f},{t_pair[1]:+.4f},{t_pair[2]:+.4f}) µm  "
              f"RMS={rms:.4f} µm ({n_corr} pts)")

        informe.append({
            'par': f'{i-1}→{i}',
            'rms_um': rms,
            'n_correspondencias': n_corr,
            'giro_eje_deg': float(euler[0]),
            'inclinacion_y_deg': float(euler[1]),
            'inclinacion_z_deg': float(euler[2]),
            'delta_um': (float(t_pair[0]), float(t_pair[1]), float(t_pair[2])),
            'arranque_frio': arranque_frio,
        })

        R_prev, t_prev = R_chain[i - 1], t_chain[i - 1]
        R_chain.append(R_prev @ R_pair)
        t_chain.append(R_prev @ t_pair + t_prev)

    # ════════════════════════════════════════════════════════════
    # CIERRE DE BUCLE
    # ════════════════════════════════════════════════════════════
    if not bucle_cerrado:
        print(f"\n[ICP] Ángulo total capturado ({angulo_total_deg:.4f}°) no es un "
              f"múltiplo de 360° -- se omite el cierre de bucle (no hay adyacencia "
              f"física entre la última y la primera captura, solo se ha capturado "
              f"una porción del cilindro).")
        resultado = []
        for i, nube in enumerate(nubes):
            p = centrar(nube)
            p_final = (R_chain[i] @ p.T).T + t_chain[i]
            p_final[:, 1] += yc
            p_final[:, 2] += zc
            resultado.append(p_final)
        _aviso_consistencia_vecinas(informe)
        return resultado, informe

    # --- Aplicar la cadena de correcciones tal cual ---
    nubes_cadena = []
    for i, nube in enumerate(nubes):
        p = centrar(nube)
        nubes_cadena.append((R_chain[i] @ p.T).T + t_chain[i])

    print("\n[ICP] Midiendo cierre de bucle real (última captura ↔ primera)...")
    src_c = nubes_cadena[-1]
    tgt_c = nubes_cadena[0]
    ang_src_c = np.arctan2(src_c[:, 2], src_c[:, 1])
    ang_tgt_c = np.arctan2(tgt_c[:, 2], tgt_c[:, 1])

    borde_c = _media_angular(
        np.array([angulos_nominales[-1], angulos_nominales[0]])
    )
    ms_c = np.abs(_dif_angular(ang_src_c, borde_c)) < vent
    mt_c = np.abs(_dif_angular(ang_tgt_c, borde_c)) < vent
    n_ms = ms_c.sum() if not isinstance(ms_c, slice) else len(src_c)
    n_mt = mt_c.sum() if not isinstance(mt_c, slice) else len(tgt_c)
    if n_ms < 20 or n_mt < 20:
        print(f"  [AVISO] Ventana de cierre con pocos puntos ({n_ms}/{n_mt}); "
              f"se usan todos los puntos (menos preciso, revisar solape real).")
        ms_c = mt_c = slice(None)

    ITER_CIERRE = max(max_iter, 400)
    R_close, t_close, rms_close, n_corr_close, arranque_frio_cierre = icp_3d(
        src_c[ms_c], tgt_c[mt_c], ITER_CIERRE, tol, dist_max,
        factor_max_relajacion, giro_max_deg,
    )
    euler_close = euler_diagnostico(R_close)

    print(f"  Cierre MEDIDO (ICP directo)      : "
          f"giro_eje={euler_close[0]:+.4f}°  "
          f"inclinación=({euler_close[1]:+.4f}°,{euler_close[2]:+.4f}°)  "
          f"Δ=({t_close[0]:+.4f},{t_close[1]:+.4f},{t_close[2]:+.4f}) µm  "
          f"RMS={rms_close:.4f} µm ({n_corr_close} pts)")

    euler_chain_final = euler_diagnostico(R_chain[-1])
    print(f"  Cierre ASUMIDO (cadena acumulada, sin corrección de cierre): "
          f"giro_eje={euler_chain_final[0]:+.4f}°  "
          f"inclinación=({euler_chain_final[1]:+.4f}°,{euler_chain_final[2]:+.4f}°)  "
          f"Δ=({t_chain[-1][0]:+.4f},{t_chain[-1][1]:+.4f},{t_chain[-1][2]:+.4f}) µm")
    R_diff = R_close @ R_chain[-1].T
    diff_ang = np.rad2deg(np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"  Discrepancia angular entre medido y asumido: {diff_ang:.4f}°  "
          f"({'la cadena era una buena aproximación' if diff_ang < 0.5 else 'diferencia notable: la cadena NO capturaba bien el cierre real'})")

    informe.append({
        'par': f'cierre({n-1}→0)',
        'rms_um': rms_close,
        'n_correspondencias': n_corr_close,
        'giro_eje_deg': float(euler_close[0]),
        'inclinacion_y_deg': float(euler_close[1]),
        'inclinacion_z_deg': float(euler_close[2]),
        'delta_um': (float(t_close[0]), float(t_close[1]), float(t_close[2])),
        'arranque_frio': arranque_frio_cierre,
    })

    # ── Repartir el cierre MEDIDO a lo largo de la cadena con SLERP ────────
    key_rots = Rotation.concatenate([Rotation.identity(), Rotation.from_matrix(R_close)])
    slerp = Slerp([0.0, 1.0], key_rots)

    resultado = []
    for i, nube in enumerate(nubes):
        f = i / (n - 1) if n > 1 else 0.0
        R_frac = slerp([f]).as_matrix()[0]
        t_frac = t_close * f   # traslación: interpolación lineal (igual que v4.1-v4.10)

        R_final = R_frac @ R_chain[i]
        t_final = R_frac @ t_chain[i] + t_frac

        p = centrar(nube)
        p_final = (R_final @ p.T).T + t_final
        p_final[:, 1] += yc
        p_final[:, 2] += zc
        resultado.append(p_final)

    _aviso_consistencia_vecinas(informe)
    return resultado, informe

# ════════════════════════════════════════════════════════════
# FUSIÓN CON PROMEDIADO
# ════════════════════════════════════════════════════════════

def fusionar(nubes: list[np.ndarray], dist: float) -> np.ndarray:
    print("\n[FUSIÓN] Promediando zonas de solapamiento...")
    total = np.vstack(nubes)
    print(f"  Total bruto: {len(total):,} pts")
    tree = cKDTree(total)
    proc = np.zeros(len(total), bool)
    out  = []
    for i in range(len(total)):
        if proc[i]: continue
        v = tree.query_ball_point(total[i], r=dist)
        out.append(total[v].mean(0))
        proc[v] = True
    res = np.array(out)
    print(f"  Tras fusión: {len(res):,} pts")
    return res

# ════════════════════════════════════════════════════════════
# ANÁLISIS TOPOGRÁFICO
# ════════════════════════════════════════════════════════════

def analizar(
    nube: np.ndarray, centro: tuple[float, float], R0: float
) -> tuple[float, np.ndarray, np.ndarray]:
    yc, zc = centro
    y = nube[:,1]-yc; z = nube[:,2]-zc
    R = np.median(np.sqrt(y**2+z**2))
    ang = np.arctan2(z, y)
    desv = np.sqrt(y**2+z**2) - R
    return R, ang, desv

def construir_mapa(
    nube: np.ndarray,
    ang_pts: np.ndarray,
    desv: np.ndarray,
    centro: tuple[float, float],
    R: float,
    n_ang: int,
    n_ax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = nube[:,0]
    bins_a = np.linspace(-np.pi, np.pi, n_ang+1)
    bins_x = np.linspace(x.min(), x.max(), n_ax+1)
    mapa = np.full((n_ang, n_ax), np.nan)
    for ia in range(n_ang):
        for ix in range(n_ax):
            m = ((ang_pts >= bins_a[ia]) & (ang_pts < bins_a[ia+1]) &
                 (x >= bins_x[ix])       & (x < bins_x[ix+1]))
            if m.sum() > 2:
                mapa[ia, ix] = desv[m].mean()
    c_ang = np.rad2deg(0.5*(bins_a[:-1]+bins_a[1:]))
    c_ax  = 0.5*(bins_x[:-1]+bins_x[1:])
    return mapa, c_ang, c_ax

# ════════════════════════════════════════════════════════════
# COLORMAP RELATIVO — SIMÉTRICO ALREDEDOR DE 0
# ════════════════════════════════════════════════════════════

def escala_simetrica(
    desv: np.ndarray, percentil: float = PERCENTIL_COLOR
) -> tuple[float, float]:

    max_abs = np.percentile(np.abs(desv[np.isfinite(desv)]), percentil)
    return -max_abs, max_abs

def desv_a_colores(
    desv_arr: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:

    t = np.clip((desv_arr - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    return cm.get_cmap(COLORMAP)(t)[:, :3].astype(np.float64)

# ════════════════════════════════════════════════════════════
# OPEN3D — MALLA 3D
# ════════════════════════════════════════════════════════════

def _normales_cilindro(
    nube: np.ndarray, centro: tuple[float, float]
) -> np.ndarray:

    yc, zc = centro
    dy = nube[:, 1] - yc
    dz = nube[:, 2] - zc
    r  = np.maximum(np.sqrt(dy**2 + dz**2), 1e-12)
    return np.column_stack([np.zeros(len(nube)), dy / r, dz / r])

def _colorear_vertices_mesh(
    mesh: "o3d.geometry.TriangleMesh",
    centro: tuple[float, float],
    R_val: float,
    vmin: float,
    vmax: float,
) -> None:

    verts = np.asarray(mesh.vertices)
    yc, zc = centro
    dy = verts[:, 1] - yc
    dz = verts[:, 2] - zc
    r_verts  = np.sqrt(dy**2 + dz**2)
    desv_v   = r_verts - R_val               # desviación radial al cilindro ideal
    colores  = desv_a_colores(desv_v, vmin, vmax)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colores)

def crear_mesh_o3d(
    nube: np.ndarray,
    centro: tuple[float, float],
    R_val: float,
    vmin: float,
    vmax: float,
) -> "o3d.geometry.TriangleMesh":

    print(f"  [Malla] Calculando normales analíticas...")
    normales = _normales_cilindro(nube, centro)

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(nube.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normales.astype(np.float64))

    if METODO_MESH == 'poisson':
        print(f"  [Malla] Reconstrucción Poisson (depth={POISSON_DEPTH})...")
        mesh, densidades = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=POISSON_DEPTH
        )
        # Eliminamos los vértices de baja densidad (caps y bordes ruidosos)
        if POISSON_TRIM > 0:
            dens = np.asarray(densidades)
            umbral = np.quantile(dens, POISSON_TRIM)
            mesh.remove_vertices_by_mask(dens < umbral)
            print(f"  [Malla] Poisson trim: {(dens < umbral).sum():,} vértices eliminados")
    else:
        print(f"  [Malla] Reconstrucción BPA (radios × {BPA_RADIOS_FACTOR})...")
        dists = pcd.compute_nearest_neighbor_distance()
        r_nn  = float(np.mean(dists))
        radios = o3d.utility.DoubleVector([r_nn * f for f in BPA_RADIOS_FACTOR])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radios
        )

    # Recortamos los artefactos en los extremos axiales (caps de Poisson)
    x_min = float(nube[:, 0].min())
    x_max = float(nube[:, 0].max())
    verts = np.asarray(mesh.vertices)
    mask_fuera = (verts[:, 0] < x_min) | (verts[:, 0] > x_max)
    if mask_fuera.any():
        mesh.remove_vertices_by_mask(mask_fuera)
        print(f"  [Malla] Recorte axial: {mask_fuera.sum():,} vértices eliminados")

    n_tri = len(mesh.triangles)
    n_v   = len(mesh.vertices)
    print(f"  [Malla] {n_v:,} vértices, {n_tri:,} triángulos")

    # Colorea la desviación del cilindro ideal
    _colorear_vertices_mesh(mesh, centro, R_val, vmin, vmax)

    # Normales para sombreado Phong
    mesh.compute_vertex_normals()

    return mesh

# ── DESARROLLO PLANO ──────────────────────────────────────────

def _construir_grid_desarrollo(
    mapa: np.ndarray,
    c_ang_deg: np.ndarray,
    c_ax: np.ndarray,
    R_val: float,
    exageracion: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n_ang, n_ax = mapa.shape
    y_arc = np.deg2rad(c_ang_deg) * R_val
    x_ax  = c_ax

    xx, yy = np.meshgrid(x_ax, y_arc)
    zz = mapa * exageracion

    mask_valido = ~np.isnan(mapa).ravel()
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    vertices = np.nan_to_num(vertices, nan=0.0)

    def idx(ia: int, ix: int) -> int:
        return ia * n_ax + ix

    triangulos = []
    for ia in range(n_ang - 1):
        for ix in range(n_ax - 1):
            v00, v01 = idx(ia, ix), idx(ia, ix + 1)
            v10, v11 = idx(ia + 1, ix), idx(ia + 1, ix + 1)
            if not (mask_valido[v00] and mask_valido[v01]
                    and mask_valido[v10] and mask_valido[v11]):
                continue
            triangulos.append((v00, v01, v10))
            triangulos.append((v01, v11, v10))

    if not triangulos:
        raise ValueError(
            "No hay datos suficientes para construir el desarrollo plano, demasiados huecos)."
        )

    return vertices, np.array(triangulos, dtype=np.int32), mask_valido

def crear_mesh_plano_o3d(
    mapa: np.ndarray,
    c_ang_deg: np.ndarray,
    c_ax: np.ndarray,
    R_val: float,
    vmin: float,
    vmax: float,
    exageracion: float = EXAGERACION_VERTICAL_DESARROLLO,
) -> "o3d.geometry.TriangleMesh":

    vertices, triangulos, mask_valido = _construir_grid_desarrollo(
        mapa, c_ang_deg, c_ax, R_val, exageracion
    )

    colores = np.full((len(vertices), 3), 0.5)  # gris neutro para celdas sin dato
    colores[mask_valido] = desv_a_colores(mapa.ravel()[mask_valido], vmin, vmax)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangulos)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colores.astype(np.float64))
    mesh.compute_vertex_normals()

    print(f"  [Malla plana] {len(vertices):,} vértices, {len(triangulos):,} triángulos "
          f"(exageración vertical ×{exageracion:g})")
    return mesh

def _crear_anillo_o3d(
    x_pos: float,
    centro: tuple[float, float],
    R_val: float,
    n: int = O3D_N_ANILLO,
) -> "o3d.geometry.LineSet":
    yc, zc = centro
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts   = np.column_stack([
        np.full(n, x_pos),
        yc + R_val * np.cos(theta),
        zc + R_val * np.sin(theta),
    ]).astype(np.float64)
    lines = [[i, (i + 1) % n] for i in range(n)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(O3D_COLOR_ANILLO, (n, 1)).astype(np.float64)
    )
    return ls

# ════════════════════════════════════════════════════════════
# OPEN3D — CLASE VISUALIZADOR
# ════════════════════════════════════════════════════════════

class _VisualizadorO3D:

    def __init__(
        self,
        mesh_cilindro: "o3d.geometry.TriangleMesh",
        mesh_plano: "o3d.geometry.TriangleMesh",
        anillo: "o3d.geometry.LineSet",
        centro: tuple[float, float],
        R_val: float,
    ) -> None:
        self._mesh_cilindro = mesh_cilindro
        self._mesh_plano    = mesh_plano
        self._anillo        = anillo
        self._centro        = centro
        self._R_val         = R_val
        self._modo          = "cilindro"   # "cilindro" | "plano"
        self._x_pendiente   = None
        self._lock          = threading.Lock()
        self._activo        = False
        self._hilo          = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def mover_corte(self, x_corte: float) -> None:

        with self._lock:
            self._x_pendiente = float(x_corte)

    @property
    def activo(self) -> bool:
        return self._activo

    def _imprimir_leyenda_comandos(self) -> None:

        print("\n  ┌─ Comandos disponibles en la ventana Open3D ──────────────")
        print("  │  Arrastrar (botón izquierdo) : Rotar")
        print("  │  Scroll                      : Zoom")
        print("  │  Arrastrar (botón derecho)    : Desplazar (pan)")
        print("  │  Tecla [Z]                   : Alternar vista "
              "cilindro completo ↔ desarrollo plano")
        print("  └───────────────────────────────────────────────────────────")

    def _alternar_vista(self, vis: "o3d.visualization.VisualizerWithKeyCallback") -> bool:

        if self._modo == "cilindro":
            vis.remove_geometry(self._mesh_cilindro, reset_bounding_box=False)
            vis.remove_geometry(self._anillo, reset_bounding_box=False)
            vis.add_geometry(self._mesh_plano, reset_bounding_box=True)
            self._modo = "plano"
            print("  [Open3D] Vista: desarrollo plano")
        else:
            vis.remove_geometry(self._mesh_plano, reset_bounding_box=False)
            vis.add_geometry(self._mesh_cilindro, reset_bounding_box=True)
            vis.add_geometry(self._anillo, reset_bounding_box=False)
            self._modo = "cilindro"
            print("  [Open3D] Vista: cilindro completo")
        return True

    def _bucle(self) -> None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name=(
                "Vista 3D — Cilindro fusionado (Open3D) "
                "| Color = desviación al cilindro ideal "
                "| Tecla Z: alternar desarrollo plano"
            ),
            width=O3D_VENTANA_ANCHO,
            height=O3D_VENTANA_ALTO,
        )

        vis.add_geometry(self._mesh_cilindro)
        vis.add_geometry(self._anillo)
        vis.register_key_callback(ord('Z'), self._alternar_vista)

        opt = vis.get_render_option()
        opt.background_color      = np.asarray(O3D_BG_COLOR, dtype=np.float64)
        opt.mesh_show_wireframe   = False
        opt.mesh_show_back_face   = True   # visible desde dentro del cilindro
        opt.show_coordinate_frame = False

        vis.reset_view_point(True)
        self._imprimir_leyenda_comandos()
        self._activo = True

        while self._activo:
            if not vis.poll_events():
                break
            with self._lock:
                x_nuevo = self._x_pendiente
                self._x_pendiente = None
            if x_nuevo is not None and self._modo == "cilindro":
                self._mover_anillo_interno(x_nuevo)
                vis.update_geometry(self._anillo)
            vis.update_renderer()

        self._activo = False
        vis.destroy_window()

    def _mover_anillo_interno(self, x_corte: float) -> None:
        yc, zc = self._centro
        n = O3D_N_ANILLO
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self._anillo.points = o3d.utility.Vector3dVector(
            np.column_stack([
                np.full(n, x_corte),
                yc + self._R_val * np.cos(theta),
                zc + self._R_val * np.sin(theta),
            ]).astype(np.float64)
        )

# ════════════════════════════════════════════════════════════
# VISUALIZACIÓN MATPLOTLIB (mapa 2D + perfil)
# ════════════════════════════════════════════════════════════

def lanzar_visualizacion(
    nube: np.ndarray,
    ang_pts: np.ndarray,
    desv: np.ndarray,
    R: float,
    centro: tuple[float, float],
    mapa: np.ndarray,
    c_ang: np.ndarray,
    c_ax: np.ndarray,
) -> None:

    # ── Escala simétrica compartida por Open3D y Matplotlib ──────────────────
    vmin, vmax = escala_simetrica(desv, PERCENTIL_COLOR)
    x_init = c_ax[len(c_ax) // 2]

    # ── 1. Arrancar Open3D ────────────────────────────────────────────────────
    vis_o3d  = None
    ax3d_mpl = None

    if _O3D_DISPONIBLE:
        print(f"\n  [Open3D] Construyendo malla del cilindro ({METODO_MESH})...")
        mesh_cilindro = crear_mesh_o3d(nube, centro, R, vmin, vmax)
        print(f"  [Open3D] Construyendo malla del desarrollo plano...")
        mesh_plano = crear_mesh_plano_o3d(
            mapa, c_ang, c_ax, R, vmin, vmax, EXAGERACION_VERTICAL_DESARROLLO
        )
        anillo = _crear_anillo_o3d(x_init, centro, R)
        vis_o3d = _VisualizadorO3D(mesh_cilindro, mesh_plano, anillo, centro, R)
        print("  [Open3D] Ventana lanzada (tecla D para alternar de vista).")
    else:
        print("  [Open3D] No disponible — fallback matplotlib 3D (punto cloud).")

    # ── 2. Tema ───────────────────────────────────────────────────────────────
    COLOR_BG    = '#0a0c12'
    COLOR_PANEL = '#111520'
    COLOR_TXT   = '#e8eaf0'
    COLOR_GRID  = '#1e2235'
    COLOR_ACC   = '#4fc3f7'
    COLOR_LINE  = '#ffd740'   # amarillo

    # ── 3. Layout matplotlib ─────────────────────────────────────────────────
    if _O3D_DISPONIBLE:
        fig = plt.figure(figsize=(11, 9), facecolor=COLOR_BG)
        fig.canvas.manager.set_window_title(
            "Análisis Confocal v4.18 — Mapa 2D y Perfil | Malla 3D en Open3D"
        )
        gs = gridspec.GridSpec(
            3, 1, figure=fig,
            left=0.09, right=0.96, top=0.93, bottom=0.08,
            hspace=0.52,
            height_ratios=[0.50, 0.40, 0.10],
        )
        ax2d    = fig.add_subplot(gs[0])
        ax_perf = fig.add_subplot(gs[1])
        ax_sli  = fig.add_subplot(gs[2])

    else:
        MAX_FALLBACK = 200_000
        idx3d = (np.random.choice(len(nube), MAX_FALLBACK, replace=False)
                 if len(nube) > MAX_FALLBACK else np.arange(len(nube)))
        fig = plt.figure(figsize=(18, 9), facecolor=COLOR_BG)
        fig.canvas.manager.set_window_title("Análisis Confocal v4.18 (fallback matplotlib 3D)")
        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            left=0.05, right=0.97, top=0.93, bottom=0.10,
            hspace=0.45, wspace=0.30,
            height_ratios=[0.48, 0.42, 0.10],
        )
        ax3d_mpl = fig.add_subplot(gs[:2, 0], projection='3d')
        ax2d     = fig.add_subplot(gs[0, 1])
        ax_perf  = fig.add_subplot(gs[1, 1])
        ax_sli   = fig.add_subplot(gs[2, 1])

        sc3d = ax3d_mpl.scatter(
            nube[idx3d,0], nube[idx3d,1], nube[idx3d,2],
            c=desv[idx3d], cmap=COLORMAP, s=1.2, alpha=0.75,
            vmin=vmin, vmax=vmax,
        )
        theta_c = np.linspace(0, 2*np.pi, 200)
        yc0, zc0 = centro
        y_r = yc0 + R * np.cos(theta_c)
        z_r = zc0 + R * np.sin(theta_c)
        line3d_cut, = ax3d_mpl.plot(
            np.full_like(theta_c, x_init), y_r, z_r,
            color=COLOR_LINE, lw=1.8, alpha=0.9, label='Corte activo',
        )
        ax3d_mpl.set_facecolor(COLOR_PANEL)
        for k in ('X','Y','Z'):
            getattr(ax3d_mpl, f'set_{k.lower()}label')(f'{k} (µm)',
                                                        color=COLOR_TXT, labelpad=6, fontsize=8)
        ax3d_mpl.set_title(
            'Vista 3D fallback — instala open3d para malla completa',
            color=COLOR_TXT, fontsize=9, pad=10,
        )
        ax3d_mpl.tick_params(colors=COLOR_TXT, labelsize=6)
        for pane in (ax3d_mpl.xaxis.pane, ax3d_mpl.yaxis.pane, ax3d_mpl.zaxis.pane):
            pane.fill = False; pane.set_edgecolor(COLOR_GRID)
        cb3d = fig.colorbar(sc3d, ax=ax3d_mpl, fraction=0.022, pad=0.12,
                            label='Desv. al cilindro ideal (µm)')
        cb3d.ax.yaxis.label.set_color(COLOR_TXT)
        cb3d.ax.tick_params(colors=COLOR_TXT, labelsize=7)

    # ── 4. Panel 1: Mapa de calor 2D ────────────
    ax2d.set_facecolor(COLOR_PANEL)
    im2d = ax2d.imshow(
        mapa, aspect='auto', origin='lower', cmap=COLORMAP,
        extent=[c_ax[0], c_ax[-1], c_ang[0], c_ang[-1]],
        vmin=vmin, vmax=vmax, interpolation='bilinear',
    )
    # Colorbar con etiqueta que deja claro que 0 = cilindro ideal
    cb2d = fig.colorbar(im2d, ax=ax2d, fraction=0.04, pad=0.02)
    cb2d.set_label('Desviación al cilindro ideal (µm)\n0 = superficie ideal',
                   color=COLOR_TXT, fontsize=7)
    cb2d.ax.tick_params(colors=COLOR_TXT, labelsize=7)
    # Marcar el 0 en la barra de color (línea blanca)
    cb2d.ax.axhline(y=0.5, color='white', lw=1.5, linestyle='--', alpha=0.8)

    vline2d = ax2d.axvline(x=x_init, color=COLOR_LINE, lw=2.0,
                            linestyle='--', alpha=0.9, label='Corte activo')
    ax2d.set_xlabel('Posición axial X (µm)', color=COLOR_TXT, fontsize=8)
    ax2d.set_ylabel('Ángulo (°)',            color=COLOR_TXT, fontsize=8)
    ax2d.set_title(
        'Mapa de calor 2D — desviación relativa al cilindro ideal\n'
        '[Rojo=por encima del ideal | Verde=ideal | Azul=por debajo]',
        color=COLOR_TXT, fontsize=9, pad=8,
    )
    ax2d.tick_params(colors=COLOR_TXT, labelsize=7)
    for sp in ax2d.spines.values(): sp.set_edgecolor(COLOR_GRID)
    ax2d.legend(fontsize=7, labelcolor=COLOR_TXT,
                facecolor='#1a1d27', edgecolor=COLOR_GRID, loc='upper right')

    # ── 5. Panel 2: Perfil de rugosidad ──────────────────────────────────────
    ax_perf.set_facecolor(COLOR_PANEL)

    def extraer_perfil(x_corte: float) -> tuple[np.ndarray, np.ndarray]:
        x = nube[:,0]
        paso_ax = (c_ax[-1] - c_ax[0]) / max(len(c_ax) - 1, 1)
        ancho   = max(paso_ax * 1.5, 0.1)
        mask    = np.abs(x - x_corte) < ancho
        if mask.sum() < 5:
            mask = np.abs(x - x_corte) < ancho * 5
        if mask.sum() == 0:
            return np.array([]), np.array([])
        a     = np.rad2deg(ang_pts[mask])
        d     = desv[mask]
        orden = np.argsort(a)
        return a[orden], d[orden]

    def calcular_metricas(d_arr: np.ndarray) -> tuple[float, float, float]:
        if len(d_arr) == 0:
            return 0.0, 0.0, 0.0
        return (np.mean(np.abs(d_arr)),
                np.sqrt(np.mean(d_arr**2)),
                float(d_arr.max() - d_arr.min()))

    ang_p0, desv_p0 = extraer_perfil(x_init)
    Ra0, Rq0, Rz0   = calcular_metricas(desv_p0)

    perf_line, = ax_perf.plot(
        ang_p0, desv_p0 * 1000 if len(desv_p0) else [],
        color=COLOR_ACC, lw=1.2, alpha=0.85,
    )
    state = {
        'fill': ax_perf.fill_between(
            ang_p0,
            desv_p0 * 1000 if len(desv_p0) else [],
            0, alpha=0.20, color=COLOR_ACC,
        )
    }
    # Línea del cilindro ideal prominente
    ax_perf.axhline(0, color=COLOR_LINE, lw=1.2, linestyle='--',
                    alpha=0.85, label='Cilindro ideal (desv = 0)')
    # Banda ±Ra para referencia visual
    if Ra0 > 0:
        ax_perf.axhspan(-Ra0 * 1000, Ra0 * 1000,
                        alpha=0.06, color='white', label=f'±Ra₀ = ±{Ra0*1e3:.3f} nm')

    txt_metrics = ax_perf.text(
        0.98, 0.95,
        f"Ra={Ra0*1000:.3f} nm\nRq={Rq0*1000:.3f} nm\nRz={Rz0*1000:.3f} nm",
        transform=ax_perf.transAxes, color=COLOR_TXT, fontsize=8,
        va='top', ha='right',
        bbox=dict(facecolor='#1a1d27', edgecolor=COLOR_GRID,
                  boxstyle='round,pad=0.4', alpha=0.85),
    )
    ax_perf.set_title(
        f'Perfil de rugosidad — corte axial X = {x_init:.3f} µm  '
        f'(0 nm = cilindro ideal)',
        color=COLOR_TXT, fontsize=9, pad=8,
    )
    ax_perf.set_xlabel('Ángulo (°)',         color=COLOR_TXT, fontsize=8)
    ax_perf.set_ylabel('Desviación (nm)',    color=COLOR_TXT, fontsize=8)
    ax_perf.tick_params(colors=COLOR_TXT, labelsize=7)
    for sp in ax_perf.spines.values(): sp.set_edgecolor(COLOR_GRID)
    ax_perf.grid(True, color=COLOR_GRID, lw=0.5, alpha=0.5)
    ax_perf.legend(fontsize=7, labelcolor=COLOR_TXT,
                   facecolor='#1a1d27', edgecolor=COLOR_GRID)

    # ── 6. Slider ─────────────────────────────────────────────────────────────
    ax_sli.set_facecolor(COLOR_BG)
    slider_cut = Slider(
        ax=ax_sli,
        label='Corte axial X (µm)',
        valmin=c_ax[0], valmax=c_ax[-1],
        valinit=x_init,
        color=COLOR_LINE,
        track_color='#1e2235',
    )
    slider_cut.label.set_color(COLOR_TXT)
    slider_cut.valtext.set_color(COLOR_TXT)
    slider_cut.poly.set_alpha(0.7)

    # ── 7. Callback ───────────────────────────────────────────────────────────
    def actualizar(val: float) -> None:
        x_corte = float(slider_cut.val)

        # Open3D: mover anillo
        if vis_o3d is not None and vis_o3d.activo:
            vis_o3d.mover_corte(x_corte)

        # Fallback 3D matplotlib
        if ax3d_mpl is not None:
            line3d_cut.set_data_3d(np.full_like(theta_c, x_corte), y_r, z_r)

        # Línea vertical en el mapa 2D
        vline2d.set_xdata([x_corte, x_corte])

        # Perfil
        a_p, d_p = extraer_perfil(x_corte)
        perf_line.set_xdata(a_p)
        perf_line.set_ydata(d_p * 1000 if len(d_p) else [])

        state['fill'].remove()
        if len(a_p) > 1:
            state['fill'] = ax_perf.fill_between(
                a_p, d_p * 1000, 0, alpha=0.20, color=COLOR_ACC
            )
        else:
            state['fill'] = ax_perf.fill_between(
                [], [], 0, alpha=0.20, color=COLOR_ACC
            )

        Ra, Rq, Rz = calcular_metricas(d_p)
        txt_metrics.set_text(
            f"Ra={Ra*1000:.3f} nm\nRq={Rq*1000:.3f} nm\nRz={Rz*1000:.3f} nm"
        )
        ax_perf.set_title(
            f'Perfil de rugosidad — corte axial X = {x_corte:.3f} µm  '
            f'(0 nm = cilindro ideal)',
            color=COLOR_TXT, fontsize=9, pad=8,
        )
        if len(d_p) > 1:
            margen = max((d_p.max() - d_p.min()) * 0.25, 0.001) * 1000
            ax_perf.set_ylim(d_p.min()*1000 - margen, d_p.max()*1000 + margen)

        fig.canvas.draw_idle()

    slider_cut.on_changed(actualizar)

    # ── 8. Botón Reset ────────────────────────────────────────────────────────
    ax_btn = fig.add_axes([0.02, 0.03, 0.07, 0.04])
    ax_btn.set_facecolor(COLOR_BG)
    btn_reset = Button(ax_btn, 'Reset corte',
                       color='#1e2235', hovercolor='#2e3555')
    btn_reset.label.set_color(COLOR_TXT)
    btn_reset.on_clicked(lambda _: slider_cut.reset())

    # ── 9. Título global ──────────────────────────────────────────────────────
    modo_3d = f"Open3D malla {METODO_MESH}" if _O3D_DISPONIBLE else "matplotlib 3D (fallback)"
    fig.suptitle(
        f"Análisis Confocal v4.18  │  R = {R:.4f} µm  │  "
        f"{len(nube):,} pts fusionados  │  {modo_3d}  │  "
        f"Escala color: [{vmin*1e3:.2f} nm … 0 … {vmax*1e3:.2f} nm]",
        color=COLOR_TXT, fontsize=10, y=0.98, fontweight='bold',
    )

    plt.show()

# ════════════════════════════════════════════════════════════
# EXPORTAR
# ════════════════════════════════════════════════════════════

def exportar(
    nube: np.ndarray,
    ang_pts: np.ndarray,
    desv: np.ndarray,
    R: float,
    centro: tuple[float, float],
    directorio: str,
    n_muestras: int,
    excentricidad_um: float,
    informe_solapes: list[dict] | None = None,
) -> None:
    yc, zc = centro
    longitud_cilindro = float(nube[:, 0].max() - nube[:, 0].min())

    # ── 1. Nube fusionada: coordenadas puras (X, Y, Z) ──────────────────────
    np.savetxt(
        os.path.join(directorio, "cilindro_fusionado_v4_18.txt"),
        nube,
        fmt="%.6f", delimiter="\t",
        header="X_um\tY_um\tZ_um", comments="",
        encoding="utf-8",
    )

    # ── 2. Desarrollo plano: mismo formato (X, Y, Z), punto a punto ─────────
    arco_real = ang_pts * R
    desarrollo = np.column_stack((nube[:, 0], arco_real, desv))
    np.savetxt(
        os.path.join(directorio, "desarrollo_plano_v4_18.txt"),
        desarrollo,
        fmt="%.6f", delimiter="\t",
        header="X_um\tY_um\tZ_um", comments="",
        encoding="utf-8",
    )

    # ── 3. Resumen ────────────────────────────────────────────────────────
    excentricidad_txt = (
        f"{excentricidad_um:.6f} µm" if not np.isnan(excentricidad_um)
        else "N/D (reajuste de centro post-fusión desactivado)"
    )
    resumen = (
        f"RESUMEN TOPOGRÁFICO\n{'='*40}\n"
        f"Nº de muestras                          : {n_muestras}\n"
        f"Radio ajustado                          : {R:.6f} µm\n"
        f"Longitud del cilindro                   : {longitud_cilindro:.6f} µm\n"
        f"Centro (Yc,Zc)                           : ({yc:.6f}, {zc:.6f}) µm\n"
        f"Excentricidad respecto al cilindro ideal : {excentricidad_txt}\n"
        f"Puntos totales                           : {len(nube):,}\n"
        f"Desv. mín.                               : {desv.min()*1000:.3f} nm\n"
        f"Desv. máx.                               : {desv.max()*1000:.3f} nm\n"
    )
    if informe_solapes:
        resumen += (
            f"\nERRORES DE SOLAPE (residuo RMS tras el ICP 3D)\n{'='*40}\n"
            f"Cuanto menor el RMS, más fielmente ha quedado encajado ese "
            f"solape.\n\n"
        )
        for info in informe_solapes:
            resumen += (
                f"Solape {info['par']:>12s} : "
                f"RMS={info['rms_um']:.4f} µm  "
                f"({info['n_correspondencias']} correspondencias)  "
                f"giro_eje={info['giro_eje_deg']:+.4f}°  "
                f"inclinación=({info['inclinacion_y_deg']:+.4f}°,"
                f"{info['inclinacion_z_deg']:+.4f}°)  "
                f"Δ=({info['delta_um'][0]:+.4f},{info['delta_um'][1]:+.4f},"
                f"{info['delta_um'][2]:+.4f}) µm"
                f"{'  [arranque en frío]' if info.get('arranque_frio') else ''}\n"
            )
            if info.get('aviso_consistencia'):
                resumen += f"    ⚠ {info['aviso_consistencia']}\n"
    with open(os.path.join(directorio, "metricas_v4_18.txt"), "w", encoding="utf-8") as f:
        f.write(resumen)
    print(resumen)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("FUSIÓN CONFOCAL v4.18 — Open3D Mesh + Matplotlib 2D")
    print("  + Unidades corregidas (µm/nm) y ventana de solape geométrica")
    print(f"  Método malla  : {METODO_MESH.upper()}")
    print(f"  Colormap      : {COLORMAP} (simétrico, 0 = cilindro ideal)")
    print("="*65)

    if not os.path.isdir(DIRECTORIO_DATOS):
        DIRECTORIO_DATOS = input(
            "Directorio no encontrado. Introduce la ruta: "
        ).strip()

    print("\n[1/7] Cargando capturas...")
    capturas = cargar_capturas(DIRECTORIO_DATOS, PATRON_ARCHIVO)

    print("\n[2/7] Estimando geometría INICIAL del cilindro (primera captura)...")
    if ESTIMAR_CENTRO:
        centro, R0 = estimar_centro(capturas, RADIO_CONOCIDO_UM,
                                    USAR_SOLO_PRIMERA_CAPTURA)
    else:
        centro = CENTRO_MANUAL
        R0     = RADIO_CONOCIDO_UM or 0.0
    print("  [AVISO] Este centro es provisional: viene de un arco corto (~1 captura)")
    print("          y solo se usa para rotar y para el ICP. Se reajustará tras fusionar.")

    print("\n[3/7] Calculando ángulos...")
    angulos = calcular_angulos(len(capturas), ANGULO_TOTAL_GRADOS, SENTIDO_GIRO)
    paso_deg = ANGULO_TOTAL_GRADOS / len(capturas)
    print(f"  Paso angular: {paso_deg:.4f}°  "
          f"(sentido={'antihorario' if SENTIDO_GIRO > 0 else 'horario'})")
    for i, (col, pts) in enumerate(capturas):
        print(f"  Captura {col}: {np.rad2deg(angulos[i]):.2f}°  ({len(pts):,} pts)")

    print("\n[4/7] Aplicando rotaciones teóricas...")
    nubes_rot = [rotar_X(pts, ang, centro)
                 for (_, pts), ang in zip(capturas, angulos)]

    if USAR_ICP and len(capturas) > 1:
        print("\n[5/7] Registro ICP...")
        nubes_reg, informe_solapes = registrar_icp(
            nubes_rot, centro, angulos, paso_deg, SOLAPAMIENTO,
            ICP_MAX_ITER, ICP_TOLERANCIA, ICP_DIST_MAX_UM, ANGULO_TOTAL_GRADOS,
            ICP_FACTOR_MAX_RELAJACION, GIRO_MAX_GRADOS,
        )
    else:
        print("\n[5/7] ICP desactivado.")
        nubes_reg = nubes_rot
        informe_solapes = []

    print("\n[6/7] Fusionando...")
    nube_final = fusionar(nubes_reg, DISTANCIA_DUPLICADOS_UM)

    # ════════════════════════════════════════════════════════════
    # REAJUSTE DE CENTRO/RADIO CON LA NUBE COMPLETA
    # ════════════════════════════════════════════════════════════
    if REAJUSTAR_CENTRO_POST_FUSION:
        print("\n[7/7] Reajustando centro/radio con la nube fusionada (360°)...")
        centro_previo = centro
        yc_new, zc_new, R_new = ajustar_circulo(nube_final[:, [1, 2]])
        print(f"  Centro previo (arco corto)  : "
              f"({centro_previo[0]:.4f}, {centro_previo[1]:.4f}) µm")
        print(f"  Centro reajustado (360°)    : "
              f"({yc_new:.4f}, {zc_new:.4f}) µm  R={R_new:.4f} µm")
        desplaz = np.hypot(yc_new - centro_previo[0], zc_new - centro_previo[1])
        print(f"  Desplazamiento del centro   : {desplaz:.4f} µm "
              f"(esto es, en esencia, la excentricidad que causaba la onda senoidal)")
        centro = (yc_new, zc_new)
        R0 = R_new
        excentricidad_um = float(desplaz)
    else:
        print("\n[7/7] Reajuste de centro post-fusión DESACTIVADO "
              "(REAJUSTAR_CENTRO_POST_FUSION=False).")
        excentricidad_um = float('nan')

    print("\n[ANÁLISIS] Calculando topografía...")
    R_fin, ang_pts, desv = analizar(nube_final, centro, R0)
    mapa, c_ang, c_ax = construir_mapa(
        nube_final, ang_pts, desv, centro, R_fin,
        N_SECCIONES_ANGULARES, N_SECCIONES_AXIALES,
    )
    Ra = np.mean(np.abs(desv))
    vmin_info, vmax_info = escala_simetrica(desv)
    print(f"  R={R_fin:.4f} µm  Ra={Ra*1e3:.3f} nm")
    print(f"  Escala de color: [{vmin_info*1e3:.3f} nm … 0 … {vmax_info*1e3:.3f} nm]")

    print("\n[EXPORT]")
    exportar(nube_final, ang_pts, desv, R_fin, centro, DIRECTORIO_DATOS,
              len(capturas), excentricidad_um, informe_solapes)

    print("\n[VIS] Lanzando visualización interactiva...")
    lanzar_visualizacion(nube_final, ang_pts, desv, R_fin, centro,
                         mapa, c_ang, c_ax)

    print("\n Programa finalizado.")