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
            output = result.stdout
            print(output)
            
            # Check if there are actual camera devices (not just system devices)
            if "camera" in output.lower() or "usb" in output.lower():
                print("\n✅ Se detectaron dispositivos de cámara USB")
            elif "pisp" in output.lower() or "hevc" in output.lower():
                print("\n⚠️ Solo se detectaron dispositivos internos del sistema")
                print("   (pisp, hevc-dec) - No son cámaras USB reales")
                print("💡 Conecta una cámara USB y verifica con: lsusb")
        else:
            print("❌ Error al ejecutar v4l2-ctl")
            print("💡 Instala con: sudo apt-get install v4l-utils")
    except FileNotFoundError:
        print("⚠️ v4l2-ctl no está instalado")
        print("💡 Instala con: sudo apt-get install v4l-utils")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Also check USB devices
    print("\n   Verificando dispositivos USB conectados...")
    try:
        result = subprocess.run(
            ["lsusb"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            usb_output = result.stdout
            if "camera" in usb_output.lower() or "webcam" in usb_output.lower() or "video" in usb_output.lower():
                print("   ✅ Se detectó un dispositivo de cámara USB:")
                for line in usb_output.split('\n'):
                    if "camera" in line.lower() or "webcam" in line.lower() or "video" in line.lower():
                        print(f"      {line}")
            else:
                print("   ⚠️ No se detectaron cámaras USB en lsusb")
                print("   💡 Conecta una cámara USB y ejecuta: lsusb")
    except Exception as e:
        print(f"   ⚠️ No se pudo ejecutar lsusb: {e}")


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
        
        # First, try to find actual camera devices (not system devices)
        import glob
        video_devices = sorted(glob.glob("/dev/video*"))
        
        # Filter out system devices (usually video19+)
        camera_devices = [d for d in video_devices if int(d.split('video')[1]) < 20]
        
        if camera_devices:
            print(f"\n   Encontrados {len(camera_devices)} dispositivo(s) de cámara potencial(es):")
            for dev in camera_devices:
                print(f"   - {dev}")
        else:
            print("\n   ⚠️ No se encontraron dispositivos de cámara en /dev/video0-19")
            print("   Los dispositivos video19+ son internos del sistema, no cámaras")
        
        # Try to open camera by index
        print("\n   Intentando abrir cámaras por índice...")
        found_camera = False
        for i in range(10):  # Try more indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ Cámara {i} funciona correctamente")
                    print(f"      - Tamaño de frame: {frame.shape}")
                    print(f"      - Puede leer frames: ✅")
                    found_camera = True
                    cap.release()
                    break
                else:
                    print(f"   ⚠️ Cámara {i} abierta pero no puede leer frames")
                    cap.release()
            else:
                cap.release()
        
        if not found_camera:
            print("   ❌ No se pudo abrir ninguna cámara funcional con OpenCV")
            print("\n   💡 Posibles soluciones:")
            print("      - Verifica que la cámara USB esté conectada")
            print("      - Prueba con: lsusb (debe mostrar la cámara)")
            print("      - Reinicia la Raspberry Pi después de conectar la cámara")
            print("      - Verifica que la cámara no esté siendo usada por otro proceso")
            
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

