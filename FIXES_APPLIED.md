# CORRECCIONES APLICADAS - Detección de Somnolencia

## 🐛 Problemas Identificados y Corregidos

### 1. ❌ PERCLOS No Funcionaba

**Problema:** El cálculo de PERCLOS estaba correcto, pero la alarma se reseteaba constantemente debido a conflictos en la lógica de reseteo.

**Solución:**

-  Eliminado el reseteo automático de alarma en `update_perclos()`
-  Movida toda la lógica de alarma a un punto centralizado en `process()`
-  Ahora se verifica que TODAS las condiciones estén claras antes de resetear la alarma

### 2. ❌ Bostezos No Se Detectaban Correctamente

**Problema:** La lógica de `update_yawn()` estaba acumulando tiempo de forma incorrecta, usando `+=` en lugar de calcular el tiempo transcurrido.

**Solución:**

-  Reescrito `update_yawn()` para calcular correctamente el tiempo transcurrido
-  Ahora guarda el tiempo de inicio y calcula la duración actual
-  La alarma se activa solo cuando el bostezo supera el umbral de tiempo

### 3. ❌ Alertas No Se Enviaban al Backend

**Problema:** Las alertas solo se enviaban en el cambio de estado `False -> True`, nunca después.

**Solución:**

-  Mejorada la lógica de envío de alertas para detectar cada evento
-  Ahora envía alertas cuando:
   -  PERCLOS supera el umbral (primera vez)
   -  Se detecta un bostezo que supera el wait_time
   -  Se detecta inclinación de cabeza que supera el wait_time
-  Prioridad: Cabeceo > Bostezo > PERCLOS

### 4. ✅ Mejoras Adicionales

-  Agregado logging detallado para debugging
-  Redondeo de valores PERCLOS para evitar errores de precisión
-  Mensajes de debug cada vez que se envía una lectura o alerta

## 🧪 Cómo Probar

### Prueba 1: PERCLOS

```bash
# Iniciar el detector
python3 auto_start.py --api-url http://tu-servidor:8000

# Mantén los ojos cerrados por más de 3-5 segundos
# Deberías ver en los logs:
# "📊 Lectura enviada - PERCLOS: 18.5%, ..."
# "🚨 Enviando alerta: SOMNOLENCIA_PERCLOS - PERCLOS: 18.5%"
```

**Resultado Esperado:**

-  PERCLOS debe aumentar gradualmente mientras los ojos están cerrados
-  Cuando PERCLOS > 15% (threshold default), se envía alerta
-  La alarma debe permanecer activa mientras PERCLOS esté alto

### Prueba 2: Bostezos

```bash
# Con el detector corriendo
# Abre la boca ampliamente (simular bostezo) por 1-2 segundos

# Deberías ver en los logs:
# "MAR: 0.75" (valor alto)
# "Yawn: 1.2s" (tiempo acumulado)
# "🚨 Enviando alerta: SOMNOLENCIA_BOSTEZOS - ... Yawn: 1.2s"
```

**Resultado Esperado:**

-  MAR debe superar 0.6 cuando abres la boca
-  El contador de yawn_time debe incrementarse
-  Cuando yawn_time >= 1.0 segundo, se envía alerta

### Prueba 3: Inclinación de Cabeza

```bash
# Con el detector corriendo
# Inclina la cabeza hacia un lado por 3-4 segundos

# Deberías ver en los logs:
# "Roll: 25.3°" (ángulo alto)
# "Tilt: 3.5s" (tiempo acumulado)
# "🚨 Enviando alerta: SOMNOLENCIA_CABECEOS - ... Tilt: 3.5s"
```

**Resultado Esperado:**

-  Roll/Pitch/Yaw debe superar los thresholds
-  El contador head_tilt_time debe incrementarse
-  Cuando head_tilt_time >= 3.0 segundos, se envía alerta

### Prueba 4: Verificar en Backend

```bash
# Verificar que las lecturas llegan
curl http://tu-servidor:8000/lecturas/?id_viaje=TU_VIAJE_ID

# Verificar que las alertas llegan
curl http://tu-servidor:8000/alertas/?id_viaje=TU_VIAJE_ID

# Deberías ver:
# - Lecturas cada 2 segundos con valores de PERCLOS, conteo_cabeceos, conteo_bostezos
# - Alertas cuando se detecta somnolencia con tipo_alerta y nivel_somnolencia
```

## 📊 Valores de Referencia

### EAR (Eye Aspect Ratio)

-  **Normal**: 0.25 - 0.35
-  **Somnoliento**: 0.18 - 0.22
-  **Cerrado**: < 0.18 (threshold default)

### MAR (Mouth Aspect Ratio)

-  **Normal**: 0.1 - 0.4
-  **Hablando**: 0.4 - 0.6
-  **Bostezo**: > 0.6 (threshold default)

### PERCLOS

-  **Alerta**: 0% - 10%
-  **Somnoliento**: 10% - 15%
-  **Peligroso**: > 15% (threshold default)
-  **Crítico**: > 20%

### Head Pose

-  **Roll** (inclinación lateral): ±20° threshold
-  **Pitch** (cabeceo adelante/atrás): ±12° threshold
-  **Yaw** (girar izquierda/derecha): ±15° threshold

## 🔍 Debugging

### Ver Logs en Tiempo Real

```bash
# Si usas auto_start.py directamente
python3 auto_start.py --api-url http://tu-servidor:8000

# Si usas el servicio systemd
sudo journalctl -u drowsiness-detector -f
```

### Logs que Deberías Ver

#### Lecturas Periódicas (cada 2 segundos)

```
📊 Lectura enviada - PERCLOS: 5.2%, EAR: 0.287, MAR: 0.234, Alarma: False
```

#### Alertas de Somnolencia

```
🚨 Enviando alerta: SOMNOLENCIA_PERCLOS - PERCLOS: 18.5%, Yawn: 0.0s, Tilt: 0.0s
🚨 Enviando alerta: SOMNOLENCIA_BOSTEZOS - PERCLOS: 8.2%, Yawn: 1.5s, Tilt: 0.0s
🚨 Enviando alerta: SOMNOLENCIA_CABECEOS - PERCLOS: 12.1%, Yawn: 0.0s, Tilt: 3.2s
```

### Si No Funciona

#### PERCLOS no aumenta cuando cierras los ojos

```bash
# Verificar threshold EAR
# Si tu EAR en reposo es < 0.18, aumenta el threshold:
python3 auto_start.py --api-url http://servidor:8000
# Edita DEFAULT_THRESHOLDS en drowsiness_detector/config.py
# Cambia EAR_THRESH de 0.18 a 0.15
```

#### Bostezos no se detectan

```bash
# Verificar threshold MAR
# Abre la boca y observa el valor MAR en los logs
# Si MAR no supera 0.6, reduce el threshold:
# Edita MAR_THRESH de 0.6 a 0.5
```

#### Alertas no llegan al backend

```bash
# Verificar conexión API
curl http://tu-servidor:8000/health

# Verificar que el viaje esté activo
curl http://tu-servidor:8000/viajes/TU_VIAJE_ID

# Verificar logs de red en el backend
# Deberías ver POST requests a /lecturas/ y /alertas/
```

## 📝 Cambios en Código

### Archivos Modificados

1. **`drowsiness_detector/detection.py`**
   -  Línea ~328: Función `update_yawn()` reescrita
   -  Línea ~320: Eliminado reseteo automático en `update_perclos()`
   -  Línea ~535: Nueva lógica centralizada de alarmas
   -  Línea ~565: Mejorado envío de alertas con prioridades

### Archivos Nuevos

2. **`auto_start.py`** - Script para ejecución automática en vehículo
3. **`drowsiness-detector.service`** - Servicio systemd
4. **`install_raspberry.sh`** - Script de instalación automatizada
5. **`RASPBERRY_PI_INSTALL.md`** - Guía completa de instalación

## ✅ Checklist de Verificación

Antes de instalar en el vehículo, verifica:

-  [ ] PERCLOS aumenta cuando cierras los ojos
-  [ ] Alarma se activa cuando PERCLOS > 15%
-  [ ] Bostezos se detectan (MAR > 0.6 por >1s)
-  [ ] Inclinación de cabeza se detecta (>3s)
-  [ ] Lecturas se envían al backend cada 2s
-  [ ] Alertas se envían al backend cuando se detectan
-  [ ] API responde correctamente (/health endpoint)
-  [ ] Viaje activo existe en el sistema
-  [ ] Cámara funciona correctamente
-  [ ] Audio de alarma suena

## 🚀 Próximos Pasos

1. **Probar en ambiente de desarrollo**

   ```bash
   cd /home/napo/Downloads/drowsiness
   source env/bin/activate
   streamlit run streamlit_app.py
   ```

2. **Verificar en consola del navegador**

   -  Abre las Developer Tools (F12)
   -  Observa la consola para errores de WebRTC o video

3. **Ajustar thresholds según tu rostro**

   -  Cada persona tiene diferentes valores de EAR/MAR
   -  Usa los sliders en la interfaz para calibrar
   -  Anota los valores que funcionan mejor

4. **Preparar para Raspberry Pi**

   ```bash
   # Copiar proyecto a Raspberry Pi
   scp -r /home/napo/Downloads/drowsiness pi@raspberrypi.local:/home/pi/

   # SSH a la Raspberry Pi
   ssh pi@raspberrypi.local

   # Ejecutar script de instalación
   cd /home/pi/drowsiness
   bash install_raspberry.sh http://TU_SERVIDOR:8000
   ```

## 🆘 Soporte

Si los problemas persisten:

1. Captura los logs completos: `sudo journalctl -u drowsiness-detector -n 200 > logs.txt`
2. Verifica la versión de las dependencias: `pip list | grep -E "(mediapipe|opencv|streamlit)"`
3. Prueba con valores de threshold más bajos/altos
4. Verifica iluminación (MediaPipe requiere buena iluminación)
