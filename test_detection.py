#!/usr/bin/env python3
"""
Script de prueba rápida para verificar detección de somnolencia.
Muestra valores en consola sin interfaz web.
"""

import cv2
import time
import sys
from pathlib import Path

# Add project to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from drowsiness_detector import VideoFrameHandler, DEFAULT_THRESHOLDS

def test_detection():
    """Test detection with webcam."""
    print("=" * 60)
    print("🧪 PRUEBA RÁPIDA - DETECTOR DE SOMNOLENCIA")
    print("=" * 60)
    print()
    print("Instrucciones:")
    print("1. Mantén los ojos cerrados por 5 segundos → Probar PERCLOS")
    print("2. Abre la boca ampliamente por 2 segundos → Probar bostezos")
    print("3. Inclina la cabeza lateralmente por 4 segundos → Probar inclinación")
    print()
    print("Presiona 'q' para salir")
    print("=" * 60)
    print()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize video handler (without viaje_id for testing)
    video_handler = VideoFrameHandler(viaje_id=None, use_raspberry_pi_optimization=False)

    # Thresholds
    thresholds = DEFAULT_THRESHOLDS.copy()

    print("✅ Cámara iniciada")
    print("📹 Procesando frames...")
    print()

    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Error leyendo frame")
                break

            # Process frame
            processed_frame, should_alarm, metrics = video_handler.process(frame, thresholds)
            frame_count += 1

            # Print status every 1 second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed

                # Status line
                status = "🚨 ALARMA" if should_alarm else "✅ Normal"

                print(f"\r{status} | "
                      f"EAR: {metrics['ear']:.3f} | "
                      f"MAR: {metrics['mar']:.3f} | "
                      f"PERCLOS: {metrics['perclos']:.1f}% | "
                      f"Yawn: {metrics['yawn_time']:.1f}s | "
                      f"Tilt: {metrics['head_tilt_time']:.1f}s | "
                      f"FPS: {fps:.1f}",
                      end='', flush=True)

                # Alert details
                if metrics.get('alert_text'):
                    print(f" | ⚠️ {metrics['alert_text']}", end='', flush=True)

                last_print_time = current_time

            # Display frame
            cv2.imshow('Drowsiness Detection Test', processed_frame)

            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️ Detenido por usuario")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n")
        print("=" * 60)
        print("📊 RESUMEN")
        print("=" * 60)
        print(f"• Frames procesados: {frame_count}")
        print(f"• Tiempo total: {elapsed:.1f}s")
        print(f"• FPS promedio: {fps:.1f}")
        print()
        print("✅ Prueba completada")


if __name__ == "__main__":
    test_detection()
