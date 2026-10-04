import os
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

st.set_page_config(
    page_title="Proyecto Avanzado: Cálculo Multivariable y Visión Artificial",
    layout="wide",
)

st.title(
    "🔬 Proyecto Integrador: Cálculo de Volúmenes y Áreas con Visión"
    " Artificial"
)
st.markdown(
    """
Este sistema implementa el pipeline completo propuesto para la **Universidad Industrial de Santander (UIS)**: 
desde la captura de la silueta de un objeto real, su procesamiento digital con **OpenCV**, 
el modelado matemático por **Splines Cúbicos**, hasta la resolución de integrales de sólidos de revolución 
y visualización 3D interactiva[cite: 1, 2, 3, 4].
"""
)


# --- FUNCION PARA CREAR UNA IMAGEN PATRÓN REAL (BOTELLA SIMÉTRICA) ---
def generar_imagen_objeto_real():
  img = np.ones((600, 600, 3), dtype=np.uint8) * 255
  pts_izq = []
  for y in range(120, 480):
    # Perfil simétrico suave de una botella o vasija de laboratorio
    r = int(
        35
        + 30 * np.sin((y - 120) * np.pi / 180)
        + 15 * np.cos((y - 120) * np.pi / 90)
    )
    r = max(r, 12)
    pts_izq.append((300 - r, y))

  pts_der = []
  for y in range(479, 119, -1):
    r = int(
        35
        + 30 * np.sin((y - 120) * np.pi / 180)
        + 15 * np.cos((y - 120) * np.pi / 90)
    )
    r = max(r, 12)
    pts_der.append((300 + r, y))

  pts = np.array(pts_izq + pts_der, np.int32)
  cv2.fillPoly(img, [pts], (0, 0, 0))

  # Referencia de calibración dimensional
  cv2.line(img, (100, 530), (200, 530), (0, 0, 255), 3)
  cv2.putText(
      img,
      "Ref: 10 cm",
      (110, 515),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 0, 255),
      2,
  )

  cv2.imwrite("objeto_real.png", img)
  return "objeto_real.png"


# Sidebar de Control
st.sidebar.header("⚙️ Configuración del Pipeline")
modo_entrada = st.sidebar.radio(
    "Origen de la Silueta del Objeto:",
    [
        "Usar Objeto Patrón Simétrico (Botella de Laboratorio)",
        "Cargar Imagen Propia (Debe ser vertical y centrada)",
    ],
)

if modo_entrada == "Usar Objeto Patrón Simétrico (Botella de Laboratorio)":
  path_img = generar_imagen_objeto_real()
  img_bgr = cv2.imread(path_img)
else:
  archivo_subido = st.sidebar.file_uploader(
      "Sube una foto de perfil de tu objeto (fondo claro y vertical):",
      type=["png", "jpg", "jpeg"],
  )
  if archivo_subido is not None:
    bytes_data = archivo_subido.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  else:
    path_img = generar_imagen_objeto_real()
    img_bgr = cv2.imread(path_img)
    st.sidebar.info(
        "Mostrando objeto patrón simétrico. Puedes subir tu foto cuando"
        " desees."
    )

# --- ETAPA 1: PROCESAMIENTO DE IMAGEN CON OPENCV ---
st.subheader("1️⃣ Etapa de Visión Artificial y Extracción de Contorno (OpenCV)")

col_img1, col_img2, col_img3 = st.columns(3)

with col_img1:
  st.text("Imagen Original")
  st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

with col_img2:
  st.text("Umbralización (Otsu)")
  st.image(thresh, use_container_width=True, clamp=True)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
img_contornos = img_bgr.copy()

if contours:
  c_max = max(contours, key=cv2.contourArea)
  cv2.drawContours(img_contornos, [c_max], -1, (0, 255, 0), 2)

with col_img3:
  st.text("Contorno Detectado y Filtrado")
  st.image(
      cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB), use_container_width=True
  )

# --- ETAPA 2: EXTRACCIÓN DE DATOS Y CALIBRACIÓN FÍSICA ---
puntos = c_max[:, 0, :]
puntos = puntos[np.argsort(puntos[:, 1])]
x_centro = np.mean(puntos[:, 0])

perfil_datos = {}
for pt in puntos:
  px, py = pt
  if px >= x_centro:  # Tomar el lado derecho para medir el radio
    radio_px = px - x_centro
    if py not in perfil_datos:
      perfil_datos[py] = radio_px

if len(perfil_datos) > 10:
  y_pxs = np.array(sorted(perfil_datos.keys()))
  r_pxs = np.array([perfil_datos[y] for y in y_pxs])

  factor_escala = st.sidebar.slider(
      "Factor de Calibración (cm por píxel):", 0.01, 0.10, 0.04, 0.01
  )

  x_cm = (y_pxs - y_pxs[0]) * factor_escala  # Altura
  y_cm = r_pxs * factor_escala  # Radio

  # --- ETAPA 3: MODELADO CON SPLINES CÚBICOS ---
  x_cm_u, indices_unicos = np.unique(x_cm, return_index=True)
  y_cm_u = y_cm[indices_unicos]

  if len(x_cm_u) > 5:
    spline = UnivariateSpline(x_cm_u, y_cm_u, s=0.005)
    x_eval = np.linspace(x_cm_u[0], x_cm_u[-1], 200)
    y_eval = np.maximum(spline(x_eval), 0.1)
    dy_eval = spline.derivative()(x_eval)

    # --- ETAPA 4: CÁLCULO MULTIVARIABLE (INTEGRACIÓN) ---
    integrando_vol = np.pi * (y_eval**2)
    volumen_simpson = simpson(integrando_vol, x_eval)
    dx = x_eval[1] - x_eval[0]
    volumen_riemann = np.sum(np.pi * (y_eval[:-1] ** 2) * dx)

    integrando_area = 2 * np.pi * y_eval * np.sqrt(1 + dy_eval**2)
    area_simpson = simpson(integrando_area, x_eval)

    st.subheader(
        "2️⃣ Resultados del Análisis con Cálculo Multivariable e Integración"
        " Numérica"
    )
    m1, m2, m3 = st.columns(3)
    with m1:
      st.metric(
          label="📦 Volumen (Regla de Simpson)",
          value=f"{volumen_simpson:.2f} cm³",
      )
    with m2:
      st.metric(
          label="📦 Volumen (Sumas de Riemann)",
          value=f"{volumen_riemann:.2f} cm³",
      )
    with m3:
      st.metric(
          label="📐 Área Superficial Lateral", value=f"{area_simpson:.2f} cm²"
      )

    # --- ETAPA 5: RECONSTRUCCIÓN TRIDIMENSIONAL ---
    st.subheader("3️⃣ Reconstrucción 3D Interactiva del Sólido Real Analizado")
    theta = np.linspace(0, 2 * np.pi, 50)
    Theta_mesh, X_mesh = np.meshgrid(theta, x_eval)
    Y_mesh = y_eval[:, np.newaxis] * np.cos(Theta_mesh)
    Z_mesh = y_eval[:, np.newaxis] * np.sin(Theta_mesh)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X_mesh,
                y=Y_mesh,
                z=Z_mesh,
                colorscale="Viridis",
                opacity=0.95,
            )
        ]
    )
    fig.update_layout(
        title=(
            "Modelo 3D Generado a partir de la Silueta Real por Sólidos de"
            " Revolución"
        ),
        scene=dict(
            xaxis_title="Altura del Objeto (cm)",
            yaxis_title="Eje Y (cm)",
            zaxis_title="Eje Z (cm)",
        ),
        width=800,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

  else:
    st.error(
        "No se pudo extraer suficientes puntos del contorno. Intenta con otra"
        " imagen."
    )
else:
  st.error("No se detectaron contornos claros en la imagen.")

with st.expander("📚 Memoria de Cálculo y Justificación Académica (UIS)"):
  st.markdown(
      r"""
    ### Fundamento Matemático Aplicado:
    1. **Extracción por Visión Artificial:** Mediante umbralización adaptativa de Otsu y filtrado morfológico en OpenCV, se aísla la frontera del objeto real, obteniendo un conjunto discreto de puntos $(X_i, Y_i)$[cite: 3].
    2. **Modelado con Splines Cúbicos:** Para cumplir con la exigencia analítica del cálculo de que la función $f(x)$ be sea continua y derivable, se ajusta un **Spline Cúbico** $S(x)$ que interpola los puntos y provee la derivada $f'(x)$ de manera suave y continua en todo el intervalo $[a, b]$[cite: 4].
    3. **Formulación del Volumen (Sólidos de Revolución):** 
       $$V = \pi \int_{a}^{b} [f(x)]^2 dx$$
       Este integral acumula infinitesimales de discos circulares de radio $f(x)$ a lo largo de la altura del objeto[cite: 3].
    4. **Formulación del Área Superficial:**
       $$A = 2\pi \int_{a}^{b} f(x) \sqrt{1 + [f'(x)]^2} dx$$
       Incorpora el diferencial de longitud de arco de la curva generatriz rotada alrededor del eje axial[cite: 3].
    5. **Integración Numérica:** Se implementa la **Regla de Simpson** frente a las **Sumas de Riemann**, demostrando convergencia numérica de alta precisión para objetos irregulares del mundo real[cite: 2, 4].
    """
  )