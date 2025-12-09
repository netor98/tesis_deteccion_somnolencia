# 🎯 PRUEBA COMPARATIVA - Streamlit vs Auto-Start

## Objetivo

Verificar que `auto_start.py` funciona **EXACTAMENTE IGUAL** que `streamlit_app.py`

## ✅ Paso 1: Probar Streamlit

```bash
cd /home/napo/Downloads/drowsiness
source env/bin/activate
streamlit run streamlit_app.py
```

### Qué observar:

1. Abre http://localhost:8501
2. Configura API URL: `http://192.168.100.82:8000`
3. Selecciona conductor o busca viaje activo
4. Observa los valores en pantalla:
   -  EAR: ~0.25-0.35 (ojos abiertos)
   -  MAR: ~0.1-0.4 (boca cerrada)
   -  PERCLOS: ~0-10% (normal)

### Pruebas:

-  **Cerrar ojos 5 segundos** → PERCLOS debe subir >15%, ver "ALERTA!!!"
-  **Abrir boca 2 segundos** → MAR >0.6, ver "BOSTEZO!!!"
-  **Inclinar cabeza 4 segundos** → Ver "CABEZA INCLINADA..."

### Anotar valores:

```
EAR normal: ______
MAR normal: ______
PERCLOS normal: ______
EAR cerrado: ______
MAR bostezo: ______
```

## ✅ Paso 2: Probar Auto-Start

```bash
# Detener Streamlit (Ctrl+C)
cd /home/napo/Downloads/drowsiness
source env/bin/activate
python3 auto_start.py --api-url http://192.168.100.82:8000 --conductor-id 1
```

### Qué observar:

Verás salida cada 5 segundos en consola:

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

### Pruebas (MISMAS que Streamlit):

-  **Cerrar ojos 5 segundos** → PERCLOS debe subir >15%, ver "🚨 ALARMA ACTIVA"
-  **Abrir boca 2 segundos** → MAR >0.6, Tiempo Bostezo >1.0s
-  **Inclinar cabeza 4 segundos** → Tiempo Cabeceo >3.0s

### Anotar valores:

```
EAR normal: ______
MAR normal: ______
PERCLOS normal: ______
EAR cerrado: ______
MAR bostezo: ______
```

## ✅ Paso 3: Comparar Resultados

Los valores deben ser **IDÉNTICOS** (±0.01 de diferencia por variación natural):

| Métrica        | Streamlit  | Auto-Start | ¿Igual? |
| -------------- | ---------- | ---------- | ------- |
| EAR normal     | **\_\_\_** | **\_\_\_** | ☐       |
| MAR normal     | **\_\_\_** | **\_\_\_** | ☐       |
| PERCLOS normal | **\_\_\_** | **\_\_\_** | ☐       |
| EAR cerrado    | **\_\_\_** | **\_\_\_** | ☐       |
| MAR bostezo    | **\_\_\_** | **\_\_\_** | ☐       |
| Alarma PERCLOS | ☐ Sí ☐ No  | ☐ Sí ☐ No  | ☐       |
| Alarma Bostezo | ☐ Sí ☐ No  | ☐ Sí ☐ No  | ☐       |
| Alarma Cabeceo | ☐ Sí ☐ No  | ☐ Sí ☐ No  | ☐       |

## ✅ Paso 4: Verificar Backend

### En Streamlit:

Abre Developer Tools (F12) > Console
Busca mensajes como:

```
📊 Lectura enviada - PERCLOS: 18.5%, EAR: 0.145, MAR: 0.256, Alarma: True
🚨 Enviando alerta: SOMNOLENCIA_PERCLOS - PERCLOS: 18.5%
```

### En Auto-Start:

Los mismos mensajes aparecen directamente en la terminal:

```
📊 Lectura enviada - PERCLOS: 18.5%, EAR: 0.145, MAR: 0.256, Alarma: True
🚨 Enviando alerta: SOMNOLENCIA_PERCLOS - PERCLOS: 18.5%, Yawn: 0.0s, Tilt: 0.0s
```

### Verificar en el Backend:

```bash
# Lecturas
curl http://192.168.100.82:8000/lecturas/?id_viaje=TU_VIAJE_ID | jq

# Alertas
curl http://192.168.100.82:8000/alertas/?id_viaje=TU_VIAJE_ID | jq
```

Ambos deben enviar:

-  ✅ Lecturas cada 2 segundos
-  ✅ Alertas cuando detectan somnolencia
-  ✅ Mismos valores de PERCLOS
-  ✅ Mismos conteos de bostezos y cabeceos

## ✅ Resultado Esperado

Si funcionan **EXACTAMENTE IGUAL**, verás:

### Valores Numéricos

-  ✅ EAR ±0.01
-  ✅ MAR ±0.01
-  ✅ PERCLOS ±0.5%
-  ✅ Tiempos de bostezo/cabeceo ±0.1s

### Comportamiento

-  ✅ Alarmas se activan en los mismos momentos
-  ✅ Backend recibe los mismos datos
-  ✅ Audio de alarma suena en ambos
-  ✅ FPS similar (±5 fps)

### Logs

```
# Streamlit (Console):
📊 Lectura enviada - PERCLOS: 18.5%, EAR: 0.145, MAR: 0.256, Alarma: True

# Auto-Start (Terminal):
📊 Lectura enviada - PERCLOS: 18.5%, EAR: 0.145, MAR: 0.256, Alarma: True

✅ IDÉNTICOS
```

## 🐛 Si NO son iguales

### EAR/MAR/PERCLOS muy diferentes

**Problema**: Cámara o iluminación cambió entre pruebas
**Solución**: Repite ambas pruebas consecutivamente, mismas condiciones

### Alarmas no se activan igual

**Problema**: Thresholds diferentes
**Solución**: Verifica que usas los mismos thresholds:

```bash
# Auto-start con mismo threshold que Streamlit
python3 auto_start.py --perclos-threshold 15.0
```

### Backend no recibe datos de Auto-Start

**Problema**: Viaje ID incorrecto o no está activo
**Solución**:

```bash
# Verificar viaje activo
curl http://192.168.100.82:8000/viajes/activos

# Usar viaje correcto
python3 auto_start.py --conductor-id ID_CORRECTO
```

### FPS muy diferente

**Problema**: Carga del sistema
**Solución**: Cerrar Streamlit antes de probar Auto-Start, y viceversa

## 🎉 Confirmación Final

Si TODOS estos checks pasan:

-  ☐ Valores EAR/MAR/PERCLOS son idénticos (±0.01)
-  ☐ Alarmas se activan en los mismos momentos
-  ☐ Backend recibe los mismos datos
-  ☐ Audio de alarma funciona en ambos
-  ☐ FPS similar en ambos
-  ☐ Logs muestran los mismos mensajes

**✅ ENTONCES AUTO-START ES 100% EQUIVALENTE A STREAMLIT**

Puedes proceder a:

1. ✅ Instalar en Raspberry Pi
2. ✅ Configurar systemd para auto-inicio
3. ✅ Instalar en vehículo

## 📝 Notas

-  **Auto-Start** es más ligero (menos RAM)
-  **Auto-Start** inicia automáticamente con systemd
-  **Auto-Start** no requiere navegador
-  **Auto-Start** usa EXACTAMENTE el mismo código de detección
-  **Auto-Start** es perfecto para Raspberry Pi en vehículo

La única diferencia es la interfaz (web vs consola), pero la lógica de detección, procesamiento y envío al backend es **100% IDÉNTICA**.
