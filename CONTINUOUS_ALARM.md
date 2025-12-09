# 🔊 Alarma Continua - Implementación

## ✅ Cambio Aplicado

La alarma ahora suena **continuamente** mientras esté activa, no solo una vez.

## 🎯 Comportamiento

### Antes (❌)
```
Alarma activada → 🔊 Suena 1 vez (2 segundos) → 🔇 Silencio
                  └─ Aunque siga en peligro, no vuelve a sonar
```

### Ahora (✅)
```
Alarma activada → 🔊 Suena → 🔊 Suena → 🔊 Suena → ... (loop continuo)
                  └─ Se repite cada 2.5 segundos hasta que se desactive
```

## 🔧 Implementación Técnica

### Auto-Start (auto_start.py)

```python
def handle_alarm(self, should_alarm):
    if should_alarm and not self.alarm_playing:
        # Inicia loop continuo en thread separado
        def play_alarm_loop():
            while not self.alarm_stop_event.is_set():
                play_alarm_sound(alarm_path, duration=2.0)
                # Pausa de 0.5s antes de repetir
                self.alarm_stop_event.wait(0.5)
        
        # Thread daemon se ejecuta en background
        self.alarm_thread = threading.Thread(target=play_alarm_loop, daemon=True)
        self.alarm_thread.start()
    
    elif not should_alarm and self.alarm_playing:
        # Detiene el loop mediante Event
        self.alarm_stop_event.set()
```

**Características:**
- ✅ Loop continuo mientras alarma esté activa
- ✅ Se repite cada 2.5 segundos (2s de audio + 0.5s de pausa)
- ✅ Thread separado no bloquea detección
- ✅ Se detiene limpiamente cuando alarma se desactiva
- ✅ Manejo de errores con reintentos

### Streamlit (streamlit_app.py)

En Streamlit no necesita cambios porque WebRTC llama a `audio_frame_callback` continuamente:

```python
def audio_frame_callback(frame: av.AudioFrame):
    play_alarm = shared_state.get("play_alarm", False)
    # AudioHandler.process() ya hace loop automático
    new_frame = audio_handler.process(frame, play_sound=play_alarm)
    return new_frame
```

**Características:**
- ✅ WebRTC llama callback ~50 veces por segundo
- ✅ `AudioHandler.process()` reproduce segmentos en loop
- ✅ Cuando `play_sound=True`, reinicia segmentos automáticamente
- ✅ Resultado: Audio continuo mientras alarma esté activa

## 📊 Ejemplo de Comportamiento

### Escenario 1: PERCLOS Alto

```
[14:30:00] PERCLOS: 18.5% → 🚨 Alarma activada
           🔊 Iniciando reproducción continua de alarma...
           
[14:30:02] 🔊 Alarma suena...
[14:30:05] 🔊 Alarma suena...
[14:30:07] 🔊 Alarma suena...
[14:30:10] 🔊 Alarma suena...
           
[14:30:15] PERCLOS: 8.2% → ✅ Estado normal
           🔇 Alarma detenida
```

### Escenario 2: Bostezo Sostenido

```
[14:35:00] MAR: 0.75, Yawn: 1.2s → 🚨 Alarma activada
           🔊 Iniciando reproducción continua de alarma...
           
[14:35:02] 🔊 Alarma suena...
[14:35:05] 🔊 Alarma suena...
[14:35:07] MAR: 0.35, Yawn: 0.0s → ✅ Estado normal
           🔇 Alarma detenida
```

### Escenario 3: Múltiples Condiciones

```
[14:40:00] PERCLOS: 16.5% + Yawn: 1.5s → 🚨 Alarma activada
           🔊 Iniciando reproducción continua de alarma...
           
[14:40:02] 🔊 Alarma suena...
[14:40:05] 🔊 Alarma suena...
[14:40:07] Yawn: 0.0s (pero PERCLOS sigue alto)
           → Alarma CONTINÚA
[14:40:10] 🔊 Alarma suena...
[14:40:12] PERCLOS: 12.1% → ✅ Estado normal
           🔇 Alarma detenida
```

## 🎚️ Parámetros Configurables

### Duración del Audio
```python
# En auto_start.py línea ~225
play_alarm_sound(alarm_path, duration=2.0)  # 2 segundos
```

### Pausa Entre Repeticiones
```python
# En auto_start.py línea ~229
self.alarm_stop_event.wait(0.5)  # 0.5 segundos de pausa
```

### Ciclo Total
```
Ciclo completo = 2.0s (audio) + 0.5s (pausa) = 2.5 segundos
Repeticiones por minuto = 60 / 2.5 = 24 veces
```

## 🔧 Ajustes Posibles

### Alarma Más Agresiva (menos pausa)
```python
# Cambiar pausa de 0.5s a 0.2s
self.alarm_stop_event.wait(0.2)
# Resultado: Suena cada 2.2s (27 veces/minuto)
```

### Alarma Más Espaciada
```python
# Cambiar pausa de 0.5s a 1.0s
self.alarm_stop_event.wait(1.0)
# Resultado: Suena cada 3.0s (20 veces/minuto)
```

### Audio Más Largo
```python
# Cambiar duración de 2.0s a 3.0s
play_alarm_sound(alarm_path, duration=3.0)
# Resultado: Suena cada 3.5s (17 veces/minuto)
```

## ⚠️ Consideraciones

### Recursos del Sistema
- Thread adicional consume CPU mínimo
- Audio se reproduce en background sin bloquear
- Se detiene limpiamente al cerrar aplicación

### Manejo de Errores
- Si falla reproducción, espera 1s antes de reintentar
- Thread daemon se termina automáticamente con programa
- Event signal permite detención limpia

### Sincronización
- Lock protege acceso a `shared_state`
- Event permite comunicación thread-safe
- No hay race conditions

## ✅ Testing

### Probar Alarma Continua

```bash
cd /home/napo/Downloads/drowsiness
source env/bin/activate
python3 auto_start.py --api-url http://servidor:8000

# Cerrar ojos por 10 segundos
# Deberías escuchar:
# 🔊 Suena (0s)
# 🔊 Suena (2.5s)
# 🔊 Suena (5s)
# 🔊 Suena (7.5s)
# 🔊 Suena (10s)
```

### Verificar que Se Detiene

```bash
# Después de cerrar ojos y activar alarma
# Abre ojos y espera
# La alarma debe detenerse cuando PERCLOS baje de 15%

# Verás en logs:
🚨 ALERTA DE SOMNOLENCIA - Activando alarma continua
🔊 Iniciando reproducción continua de alarma...
[varios ciclos de audio]
✅ Estado normal - Desactivando alarma
🔇 Alarma detenida
```

## 🎉 Resumen

✅ **Alarma ahora es continua** - Suena repetidamente mientras haya peligro
✅ **Se detiene automáticamente** - Cuando condiciones vuelven a normal
✅ **No bloquea detección** - Thread separado permite operación paralela
✅ **Limpieza apropiada** - Se detiene correctamente al cerrar programa
✅ **Consistente con Streamlit** - Ambas versiones tienen comportamiento similar

**Resultado**: El conductor será alertado continuamente hasta que corrija la situación de somnolencia.
