#!/usr/bin/env python3
"""
Script de diagnóstico para verificar el acceso a la cámara en Raspberry Pi.
Ejecutar con: python check_camera.py
"""

import sys
import os

def check_v4l2_devices():
    """Verifica dispositivos de video disponibles usando v4l2."""
    print("=" * 60)
    print("1. Verificando dispositivos de video (v4l2)...")
    print("=" * 60)
    
    import subprocess
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Error al ejecutar v4l2-ctl")
            print("💡 Instala con: sudo apt-get install v4l-utils")
    except FileNotFoundError:
        print("⚠️ v4l2-ctl no está instalado")
        print("💡 Instala con: sudo apt-get install v4l-utils")
    except Exception as e:
        print(f"❌ Error: {e}")


def check_video_devices():
    """Verifica dispositivos /dev/video* disponibles."""
    print("\n" + "=" * 60)
    print("2. Verificando dispositivos /dev/video*...")
    print("=" * 60)
    
    import glob
    video_devices = glob.glob("/dev/video*")
    if video_devices:
        print(f"✅ Encontrados {len(video_devices)} dispositivo(s) de video:")
        for device in sorted(video_devices):
            print(f"   - {device}")
            # Check permissions
            if os.access(device, os.R_OK):
                print(f"     ✅ Lectura permitida")
            else:
                print(f"     ❌ Sin permiso de lectura")
    else:
        print("❌ No se encontraron dispositivos /dev/video*")
        print("💡 Verifica que la cámara esté conectada")


def check_opencv():
    """Verifica si OpenCV puede acceder a la cámara."""
    print("\n" + "=" * 60)
    print("3. Verificando acceso con OpenCV...")
    print("=" * 60)
    
    try:
        import cv2
        print("✅ OpenCV está instalado")
        
        # Try to open camera
        for i in range(3):
            print(f"\n   Intentando abrir /dev/video{i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"   ✅ Cámara {i} abierta exitosamente")
                ret, frame = cap.read()
                if ret:
                    print(f"   ✅ Puede leer frames (tamaño: {frame.shape})")
                else:
                    print(f"   ⚠️ Abierta pero no puede leer frames")
                cap.release()
                break
            else:
                print(f"   ❌ No se pudo abrir cámara {i}")
                cap.release()
        else:
            print("   ❌ No se pudo abrir ninguna cámara con OpenCV")
            
    except ImportError:
        print("❌ OpenCV no está instalado")
        print("💡 Instala con: pip install opencv-python")
    except Exception as e:
        print(f"❌ Error: {e}")


def check_permissions():
    """Verifica permisos del usuario."""
    print("\n" + "=" * 60)
    print("4. Verificando permisos del usuario...")
    print("=" * 60)
    
    import getpass
    user = getpass.getuser()
    print(f"Usuario actual: {user}")
    
    # Check if user is in video group
    import grp
    try:
        video_group = grp.getgrnam('video')
        if user in video_group.gr_mem:
            print("✅ Usuario está en el grupo 'video'")
        else:
            print("⚠️ Usuario NO está en el grupo 'video'")
            print("💡 Agrega con: sudo usermod -a -G video $USER")
            print("   Luego cierra sesión y vuelve a iniciar sesión")
    except KeyError:
        print("⚠️ Grupo 'video' no existe")


def check_streamlit_webrtc():
    """Verifica si streamlit-webrtc está instalado."""
    print("\n" + "=" * 60)
    print("5. Verificando dependencias...")
    print("=" * 60)
    
    try:
        import streamlit_webrtc
        print("✅ streamlit-webrtc está instalado")
    except ImportError:
        print("❌ streamlit-webrtc no está instalado")
        print("💡 Instala con: pip install streamlit-webrtc")
    
    try:
        import av
        print("✅ av (PyAV) está instalado")
    except ImportError:
        print("❌ av (PyAV) no está instalado")
        print("💡 Instala con: pip install av")


def main():
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO DE CÁMARA PARA RASPBERRY PI")
    print("=" * 60 + "\n")
    
    check_v4l2_devices()
    check_video_devices()
    check_permissions()
    check_opencv()
    check_streamlit_webrtc()
    
    print("\n" + "=" * 60)
    print("RESUMEN Y RECOMENDACIONES")
    print("=" * 60)
    print("""
NOTA IMPORTANTE:
- streamlit-webrtc usa la cámara del NAVEGADOR, no del servidor
- Si accedes desde otra máquina, el navegador usará la cámara de esa máquina
- Si accedes localmente en la Raspberry Pi, el navegador usará la cámara de la Pi

Si el error persiste:
1. Verifica que el navegador tenga permisos para acceder a la cámara
2. Asegúrate de acceder con http:// o https:// (no file://)
3. Prueba con diferentes navegadores (Chrome, Firefox)
4. Verifica la configuración de privacidad del navegador
    """)


if __name__ == "__main__":
    main()

