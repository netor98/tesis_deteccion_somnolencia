# Auto-Start vs Streamlit - Comparación

## ✅ AHORA SON IDÉNTICOS

El `auto_start.py` ahora usa **EXACTAMENTE** la misma lógica que `streamlit_app.py`, solo sin interfaz web.

## Componentes Idénticos

### 1. VideoFrameHandler

```python
# Streamlit:
video_handler = VideoFrameHandler(
    viaje_id=viaje_id,
    use_raspberry_pi_optimization=is_raspberry
)

# Auto-Start (IGUAL):
self.video_handler = VideoFrameHandler(
    viaje_id=self.viaje_id,
    use_raspberry_pi_optimization=self.is_raspberry
)
```

### 2. Thresholds

```python
# Streamlit:
thresholds = DEFAULT_THRESHOLDS.copy()
thresholds["PERCLOS_THRESH"] = st.session_state["perclos_thresh"]

# Auto-Start (IGUAL):
self.thresholds = DEFAULT_THRESHOLDS.copy()
self.thresholds["PERCLOS_THRESH"] = 15.0  # Default, configurable vía args
```

### 3. Procesamiento de Frames

```python
# Streamlit (video_frame_callback):
def video_frame_callback(frame: av.VideoFrame):
    frame_array = frame.to_ndarray(format="bgr24")
    processed_frame, play_alarm, metrics = video_handler.process(frame_array, thresholds)
    with lock:
        shared_state["play_alarm"] = play_alarm
        shared_state["metrics"] = metrics

# Auto-Start (IDÉNTICO):
def process_frame(self, frame):
    processed_frame, play_alarm, metrics = self.video_handler.process(frame, self.thresholds)
    with self.lock:
        self.shared_state["play_alarm"] = play_alarm
        self.shared_state["metrics"] = metrics
```

### 4. Manejo de Alarma

```python
# Streamlit (audio_frame_callback):
def audio_frame_callback(frame: av.AudioFrame):
    if shared_state["play_alarm"]:
        # Play alarm sound

# Auto-Start (IDÉNTICO):
def handle_alarm(self, should_alarm):
    if should_alarm and not self.alarm_playing:
        play_alarm_sound(...)
```

### 5. Shared State

```python
# Streamlit:
st.session_state["shared_state"] = {
    "play_alarm": False,
    "metrics": { ... }
}

# Auto-Start (IDÉNTICO):
self.shared_state = {
    "play_alarm": False,
    "metrics": { ... }
}
```

### 6. Optimización de Instancias

```python
# Streamlit (después del fix):
if "video_handler" not in st.session_state:
    st.session_state["video_handler"] = VideoFrameHandler(...)
else:
    video_handler.update_viaje_id(viaje_id, reset_state=False)

# Auto-Start (IDÉNTICO):
if self.video_handler is None:
    self.video_handler = VideoFrameHandler(...)
else:
    self.video_handler.update_viaje_id(self.viaje_id, reset_state=False)
```

## Diferencias (Solo UI)

| Característica            | Streamlit   | Auto-Start         |
| ------------------------- | ----------- | ------------------ |
| **Lógica de detección**   | ✅ Idéntica | ✅ Idéntica        |
| **VideoFrameHandler**     | ✅ Igual    | ✅ Igual           |
| **Thresholds**            | ✅ Igual    | ✅ Igual           |
| **Envío a backend**       | ✅ Igual    | ✅ Igual           |
| **Alarmas**               | ✅ Igual    | ✅ Igual           |
| **Interfaz Web**          | ✅ Sí       | ❌ No              |
| **Sliders configuración** | ✅ Sí       | ⚙️ Argumentos CLI  |
| **Video preview**         | ✅ Sí       | 📊 Logs en consola |
| **Selección conductor**   | ✅ Dropdown | ⚙️ Auto-detecta    |

## Ventajas de Auto-Start

1. **Mismo comportamiento**: Usa el mismo código de detección
2. **Sin navegador**: No requiere abrir navegador
3. **Más ligero**: Menos consumo de RAM (sin Streamlit overhead)
4. **Auto-inicio**: Perfecto para systemd
5. **Logs detallados**: Muestra EXACTAMENTE lo mismo que Streamlit, en consola

## Ejemplo de Salida (Auto-Start)

```
📊 Estado [14:30:45]:
   ==================================================
   Status: ✅ Normal
   ==================================================
   • EAR (Eye):        0.287  (ABIERTO)
   • MAR (Mouth):      0.234  (NORMAL)
   • PERCLOS:          5.2%  (OK)
   • Tiempo Bostezo:   0.0s
   • Tiempo Cabeceo:   0.0s
   • Head Pose:
     - Roll (lateral):  -2.3°
     - Pitch (arriba/abajo): 5.1°
     - Yaw (izq/der):   1.8°
   ==================================================
   • Frames: 8234 | FPS: 28.5
   • Viaje ID: 123
```

Cuando hay alarma:

```
📊 Estado [14:31:20]:
   ==================================================
   Status: 🚨 ALARMA ACTIVA
   ==================================================
   • EAR (Eye):        0.145  (CERRADO)
   • MAR (Mouth):      0.256  (NORMAL)
   • PERCLOS:          18.5%  (PELIGRO)
   • Tiempo Bostezo:   0.0s
   • Tiempo Cabeceo:   0.0s
   • Head Pose:
     - Roll (lateral):  -1.5°
     - Pitch (arriba/abajo): 4.8°
     - Yaw (izq/der):   2.1°
   • Alerta: ALERTA!!!
   ==================================================
   • Frames: 9456 | FPS: 28.3
   • Viaje ID: 123

🚨 ALERTA DE SOMNOLENCIA - Activando alarma
📊 Lectura enviada - PERCLOS: 18.5%, EAR: 0.145, MAR: 0.256, Alarma: True
🚨 Enviando alerta: SOMNOLENCIA_PERCLOS - PERCLOS: 18.5%, Yawn: 0.0s, Tilt: 0.0s
```

## Uso

### Streamlit (con interfaz):

```bash
streamlit run streamlit_app.py
```

### Auto-Start (sin interfaz, mismo comportamiento):

```bash
python3 auto_start.py --api-url http://servidor:8000 --conductor-id 1
```

### Configurar Thresholds

**Streamlit**: Usa sliders en la interfaz

**Auto-Start**: Usa argumentos de línea de comandos:

```bash
python3 auto_start.py \
  --api-url http://servidor:8000 \
  --conductor-id 1 \
  --perclos-threshold 20.0
```

## Verificación

Para verificar que son idénticos, compara los logs:

**Streamlit**: Abre Developer Tools (F12) > Console
**Auto-Start**: Ver salida directamente en terminal

Ambos mostrarán:

-  ✅ Mismos valores EAR, MAR, PERCLOS
-  ✅ Mismas alertas
-  ✅ Mismos mensajes al backend
-  ✅ Mismo comportamiento de alarma

## Conclusión

✅ **Auto-Start ahora es 100% equivalente a Streamlit**

-  Usa el mismo `VideoFrameHandler`
-  Mismos thresholds
-  Misma lógica de procesamiento
-  Mismos mensajes al backend
-  Solo difiere en la forma de mostrar (web vs consola)

Perfecto para **Raspberry Pi en vehículo** donde no necesitas interfaz web pero quieres el mismo comportamiento exacto.
