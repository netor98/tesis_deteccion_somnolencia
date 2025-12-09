# Guía de Instalación - Raspberry Pi en Vehículo

Esta guía explica cómo instalar y configurar el sistema de detección de somnolencia en una Raspberry Pi que estará instalada permanentemente en un vehículo.

## 📋 Requisitos

### Hardware

-  Raspberry Pi 4 (4GB RAM recomendado)
-  Cámara USB o Raspberry Pi Camera Module
-  Altavoz o sistema de audio del vehículo
-  Tarjeta microSD (32GB mínimo)
-  Fuente de alimentación para el carro (12V a 5V)

### Red

-  Router WiFi en el vehículo O
-  Hotspot móvil 4G/5G O
-  Conexión directa al servidor backend

## 🚀 Instalación Inicial

### 1. Preparar Raspberry Pi

```bash
# Actualizar sistema
sudo apt-get update
sudo apt-get upgrade -y

# Instalar dependencias del sistema
sudo apt-get install -y python3-pip python3-venv git
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt-get install -y libharfbuzz0b libwebp7 libtiff5 libjasper-dev
sudo apt-get install -y libqtgui4 libqt4-test libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y alsa-utils # Para reproducir audio

# Verificar cámara
ls /dev/video*
# Debería mostrar /dev/video0 o similar
```

### 2. Clonar e Instalar Proyecto

```bash
# Navegar a home
cd /home/pi

# Clonar o copiar el proyecto
git clone <tu-repositorio-url> drowsiness
# O si copias manualmente:
# scp -r /ruta/local/drowsiness pi@raspberrypi.local:/home/pi/

cd drowsiness

# Crear entorno virtual
python3 -m venv env

# Activar entorno
source env/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Verificar instalación
python3 -c "import cv2, mediapipe; print('OK')"
```

### 3. Configurar Variables de Entorno

Editar o crear archivo de configuración:

```bash
nano /home/pi/drowsiness/config.env
```

Agregar:

```bash
# URL del backend API (IMPORTANTE: cambiar a tu servidor)
RISK_ADVISOR_API_URL=http://192.168.100.82:8000

# O si usas dominio:
# RISK_ADVISOR_API_URL=https://tu-dominio.com/api

# ID del conductor (opcional, se auto-detectará)
# CONDUCTOR_ID=1

# Índice de la cámara
CAMERA_INDEX=0

# Umbral PERCLOS (sensibilidad de detección)
PERCLOS_THRESHOLD=15.0
```

### 4. Probar Manualmente

Antes de configurar el auto-inicio, probar que funciona:

```bash
cd /home/pi/drowsiness
source env/bin/activate

# Probar con conductor específico
python3 auto_start.py --conductor-id 1 --api-url http://192.168.100.82:8000

# O dejar que auto-detecte
python3 auto_start.py --api-url http://192.168.100.82:8000
```

Deberías ver:

```
🚗 DETECTOR DE SOMNOLENCIA - MODO AUTOMÁTICO
============================================================
🔌 Esperando conexión con API: http://192.168.100.82:8000
✅ Conectado a la API en intento 1
🔍 Buscando viaje activo para conductor ID: 1
✅ Viaje activo encontrado: #123
📷 Inicializando cámara (índice: 0)...
✅ Cámara inicializada
🎥 Inicializando procesador de video...
✅ Procesador de video inicializado
🎬 INICIANDO DETECCIÓN
```

Si funciona correctamente, presiona `Ctrl+C` para detener.

## 🔄 Configurar Auto-Inicio

### 1. Editar Archivo de Servicio

```bash
# Editar el archivo de servicio con tus rutas
nano /home/pi/drowsiness/drowsiness-detector.service
```

Actualizar las siguientes líneas según tu configuración:

```ini
User=pi  # Cambiar si usas otro usuario

WorkingDirectory=/home/pi/drowsiness  # Ruta del proyecto

Environment="RISK_ADVISOR_API_URL=http://TU_IP:8000"  # TU URL DE API

ExecStart=/home/pi/drowsiness/env/bin/python3 /home/pi/drowsiness/auto_start.py --api-url http://TU_IP:8000
```

### 2. Instalar Servicio

```bash
# Copiar servicio a systemd
sudo cp /home/pi/drowsiness/drowsiness-detector.service /etc/systemd/system/

# Recargar systemd
sudo systemctl daemon-reload

# Habilitar servicio para auto-inicio
sudo systemctl enable drowsiness-detector

# Iniciar servicio
sudo systemctl start drowsiness-detector

# Verificar estado
sudo systemctl status drowsiness-detector
```

Deberías ver:

```
● drowsiness-detector.service - Driver Drowsiness Detection System
     Loaded: loaded (/etc/systemd/system/drowsiness-detector.service; enabled)
     Active: active (running) since ...
```

### 3. Ver Logs en Tiempo Real

```bash
# Ver logs del servicio
sudo journalctl -u drowsiness-detector -f

# O los últimos 100 líneas
sudo journalctl -u drowsiness-detector -n 100
```

## 🔧 Comandos de Mantenimiento

```bash
# Ver estado del servicio
sudo systemctl status drowsiness-detector

# Detener servicio
sudo systemctl stop drowsiness-detector

# Reiniciar servicio
sudo systemctl restart drowsiness-detector

# Deshabilitar auto-inicio
sudo systemctl disable drowsiness-detector

# Ver logs
sudo journalctl -u drowsiness-detector -f

# Limpiar logs antiguos
sudo journalctl --vacuum-time=7d
```

## 📱 Configuración de Red

### Opción 1: WiFi del Vehículo

Si el vehículo tiene su propio router WiFi:

```bash
# Configurar WiFi
sudo raspi-config
# Seleccionar: System Options > Wireless LAN
# Ingresar SSID y contraseña

# Verificar conexión
ping -c 4 8.8.8.8
```

### Opción 2: Hotspot Móvil

Si usas hotspot del celular o dispositivo 4G:

```bash
# Conectar al hotspot manualmente una vez
# La Raspberry recordará la red

# Para conexión automática, editar:
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

Agregar:

```
network={
    ssid="NombreDelHotspot"
    psk="ContraseñaDelHotspot"
    key_mgmt=WPA-PSK
    priority=10
}
```

### Opción 3: Red Directa con Backend

Si el backend está en el mismo vehículo o red local:

-  Asignar IP estática a la Raspberry Pi
-  Configurar el servidor backend para escuchar en `0.0.0.0`
-  Usar IP local en `RISK_ADVISOR_API_URL`

## 🎯 Flujo de Operación en el Vehículo

### Al Encender el Vehículo

1. **Raspberry Pi enciende** automáticamente (conectada al sistema eléctrico)
2. **Servicio inicia** después de 30 segundos (tiempo de espera configurado)
3. **Busca conexión** con el backend API
   -  Si no hay conexión, reintenta cada 5 segundos por 5 minutos
4. **Busca viaje activo** para el conductor
   -  Si no hay viaje, espera y verifica cada 10 segundos
5. **Inicia detección** automáticamente cuando encuentra viaje activo
6. **Envía datos** al backend cada 2 segundos
7. **Activa alarma** cuando detecta somnolencia

### Durante el Viaje

-  Detección continua sin intervención
-  Datos enviados automáticamente al backend
-  Alarmas se activan automáticamente
-  Sistema se recupera automáticamente de errores

### Al Apagar el Vehículo

-  La Raspberry Pi se apaga con el vehículo (usar protección contra apagado brusco)
-  El servicio guarda estado y termina limpiamente

## 🔐 Configuración de Conductores

### Método 1: Asignación Manual

Editar el servicio para incluir ID de conductor específico:

```bash
sudo nano /etc/systemd/system/drowsiness-detector.service
```

Cambiar la línea `ExecStart`:

```ini
ExecStart=/home/pi/drowsiness/env/bin/python3 /home/pi/drowsiness/auto_start.py --conductor-id 1 --api-url http://TU_IP:8000
```

### Método 2: Auto-Detección

Dejar sin `--conductor-id` y el sistema usará el primer conductor disponible.

### Método 3: Archivo de Configuración por Vehículo

Crear archivo de configuración:

```bash
nano /home/pi/drowsiness/vehicle_config.txt
```

Contenido:

```
CONDUCTOR_ID=1
VEHICLE_ID=ABC123
```

Modificar `auto_start.py` para leer este archivo.

## 📊 Monitoreo y Mantenimiento

### Verificar que está Funcionando

```bash
# Desde otra computadora en la misma red
ssh pi@<IP_RASPBERRY>

# Ver estado
sudo systemctl status drowsiness-detector

# Ver logs
sudo journalctl -u drowsiness-detector -f
```

### Logs a Revisar

Los logs muestran cada 30 segundos:

-  Frames procesados
-  FPS (ideal: 15-30 en Raspberry Pi)
-  Métricas EAR, MAR, PERCLOS
-  Estado de alarma

### Problemas Comunes

#### ❌ "No se pudo abrir la cámara"

```bash
# Verificar cámara
ls /dev/video*
v4l2-ctl --list-devices

# Cambiar índice si es necesario
python3 auto_start.py --camera 1
```

#### ❌ "No se puede conectar a la API"

```bash
# Verificar red
ping <IP_DEL_BACKEND>

# Verificar backend está corriendo
curl http://<IP_DEL_BACKEND>:8000/health

# Ver URL configurada
grep RISK_ADVISOR_API_URL /etc/systemd/system/drowsiness-detector.service
```

#### ❌ "No hay viaje activo"

-  Verificar que existe un conductor en el sistema
-  Iniciar un viaje desde el panel de administración
-  El sistema esperará hasta que haya un viaje activo

#### ⚠️ FPS muy bajo (<5)

```bash
# Reducir resolución en auto_start.py
# Editar líneas 136-138 para usar resolución más baja:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

## 🔒 Seguridad y Protección

### Protección contra Apagado Brusco

Instalar protección de tarjeta SD:

```bash
# Habilitar boot-time overlay para solo-lectura parcial
sudo raspi-config
# Advanced Options > Overlay File System
```

### Backup de Configuración

```bash
# Hacer backup de la configuración
tar -czf drowsiness-backup.tar.gz /home/pi/drowsiness /etc/systemd/system/drowsiness-detector.service

# Copiar a servidor remoto
scp drowsiness-backup.tar.gz usuario@servidor:/backups/
```

## 📈 Optimización de Rendimiento

```bash
# Deshabilitar interfaz gráfica (libera RAM)
sudo systemctl set-default multi-user.target

# Aumentar swap para Mediapipe
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Cambiar CONF_SWAPSIZE=100 a CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Sobreclock (solo si tienes buena ventilación)
sudo nano /boot/config.txt
# Agregar:
# over_voltage=2
# arm_freq=1750
```

## 🎓 Resumen del Sistema

```
ENCENDIDO → ESPERA CONEXIÓN API → ESPERA VIAJE ACTIVO → INICIA DETECCIÓN
    ↓              ↓                     ↓                      ↓
AUTOMÁTICO     REINTENTA          VERIFICA CADA        ENVÍA DATOS
               (5s x 5min)        10 SEGUNDOS          CADA 2s
                                                            ↓
                                                    ALARMA SI SOMNOLENCIA
```

¡El sistema está diseñado para funcionar completamente sin intervención humana!

## 📞 Soporte

Si encuentras problemas, revisa los logs:

```bash
sudo journalctl -u drowsiness-detector --since "1 hour ago"
```

Y contacta al equipo de desarrollo con los detalles del error.
