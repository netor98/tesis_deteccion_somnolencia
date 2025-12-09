#!/bin/bash
# Script de instalación rápida para Raspberry Pi
# Ejecutar con: bash install_raspberry.sh

set -e  # Salir si hay error

echo "=================================="
echo "🚗 Instalación Detector Somnolencia"
echo "=================================="
echo ""

# Variables configurables
API_URL="${1:-http://192.168.100.82:8000}"
INSTALL_DIR="/home/pi/drowsiness"
CONDUCTOR_ID="${2:-}"

echo "📋 Configuración:"
echo "   • URL API: $API_URL"
echo "   • Directorio: $INSTALL_DIR"
if [ -n "$CONDUCTOR_ID" ]; then
    echo "   • Conductor ID: $CONDUCTOR_ID"
else
    echo "   • Conductor ID: Auto-detectar"
fi
echo ""

read -p "¿Continuar con la instalación? (s/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "❌ Instalación cancelada"
    exit 1
fi

echo ""
echo "📦 Paso 1/5: Actualizando sistema..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

echo ""
echo "📦 Paso 2/5: Instalando dependencias del sistema..."
sudo apt-get install -y -qq \
    python3-pip python3-venv git \
    libatlas-base-dev libhdf5-dev \
    libharfbuzz0b libwebp7 libtiff5 \
    alsa-utils \
    v4l-utils

echo ""
echo "📦 Paso 3/5: Instalando paquetes Python..."
cd "$INSTALL_DIR" || { echo "❌ Error: Directorio $INSTALL_DIR no existe"; exit 1; }

if [ ! -d "env" ]; then
    python3 -m venv env
fi

source env/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "🎥 Paso 4/5: Verificando cámara..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✅ Cámara detectada: $(ls /dev/video*)"
else
    echo "⚠️  Advertencia: No se detectó cámara USB"
    echo "   Conecta una cámara USB o habilita la cámara de Raspberry Pi"
fi

echo ""
echo "🔧 Paso 5/5: Configurando servicio systemd..."

# Crear archivo de servicio con configuración
SERVICE_FILE="/tmp/drowsiness-detector.service"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Driver Drowsiness Detection System
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="RISK_ADVISOR_API_URL=$API_URL"
Environment="PYTHONUNBUFFERED=1"
ExecStartPre=/bin/sleep 30
EOF

# Agregar comando de inicio con o sin conductor_id
if [ -n "$CONDUCTOR_ID" ]; then
    echo "ExecStart=$INSTALL_DIR/env/bin/python3 $INSTALL_DIR/auto_start.py --conductor-id $CONDUCTOR_ID --api-url $API_URL" >> "$SERVICE_FILE"
else
    echo "ExecStart=$INSTALL_DIR/env/bin/python3 $INSTALL_DIR/auto_start.py --api-url $API_URL" >> "$SERVICE_FILE"
fi

cat >> "$SERVICE_FILE" << EOF
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=drowsiness-detector
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Copiar y activar servicio
sudo cp "$SERVICE_FILE" /etc/systemd/system/drowsiness-detector.service
sudo systemctl daemon-reload
sudo systemctl enable drowsiness-detector

echo ""
echo "=================================="
echo "✅ ¡Instalación Completada!"
echo "=================================="
echo ""
echo "📝 Próximos pasos:"
echo ""
echo "1. Verificar conexión con API:"
echo "   curl $API_URL/health"
echo ""
echo "2. Probar manualmente (opcional):"
echo "   cd $INSTALL_DIR"
echo "   source env/bin/activate"
echo "   python3 auto_start.py --api-url $API_URL"
echo ""
echo "3. Iniciar servicio automático:"
echo "   sudo systemctl start drowsiness-detector"
echo ""
echo "4. Ver estado y logs:"
echo "   sudo systemctl status drowsiness-detector"
echo "   sudo journalctl -u drowsiness-detector -f"
echo ""
echo "5. Para reiniciar Raspberry Pi:"
echo "   sudo reboot"
echo ""
echo "🎯 El sistema se iniciará automáticamente al encender"
echo ""
