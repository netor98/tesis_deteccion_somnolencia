#!/usr/bin/env python3
"""
Standalone Drowsiness Detection - No Browser Required

This script runs drowsiness detection directly using OpenCV to capture
from the local camera, without needing a web browser.

Usage:
    python standalone_detector.py --viaje_id 123
    python standalone_detector.py --viaje_id 123 --camera 0
    python standalone_detector.py --viaje_id 123 --headless  # No window display
"""

import os
import sys
import cv2
import time
import argparse
import signal
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drowsiness_detector import VideoFrameHandler, DEFAULT_THRESHOLDS, get_api_client
from drowsiness_detector.api_client import send_reading_async, send_alert_async

# Try to import audio support
try:
    from pydub import AudioSegment
    from pydub.playback import play
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  pydub no disponible - las alarmas de audio estarán deshabilitadas")


class StandaloneDetector:
    """Standalone drowsiness detector that runs without a browser."""
    
    def __init__(self, viaje_id: int, camera_id: int = 0, headless: bool = False,
                 api_url: str = "http://localhost:8000", show_metrics: bool = True):
        """
        Initialize the standalone detector.
        
        Args:
            viaje_id: ID of the active trip
            camera_id: Camera device ID (0 for default camera)
            headless: If True, run without displaying video window
            api_url: Backend API URL
            show_metrics: If True, print metrics to console
        """
        self.viaje_id = viaje_id
        self.camera_id = camera_id
        self.headless = headless
        self.api_url = api_url
        self.show_metrics = show_metrics
        self.running = False
        
        # Initialize video handler
        self.video_handler = VideoFrameHandler(viaje_id=viaje_id)
        
        # Set up API client
        api_client = get_api_client()
        api_client.set_base_url(api_url)
        
        # Thresholds (can be customized)
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        
        # Audio alarm
        self.alarm_file = os.path.join(os.path.dirname(__file__), "audio", "wake_up.wav")
        self.alarm_sound = None
        self.alarm_playing = False
        self.last_alarm_time = 0
        self.alarm_cooldown = 3.0  # Seconds between alarms
        
        if AUDIO_AVAILABLE and os.path.exists(self.alarm_file):
            try:
                self.alarm_sound = AudioSegment.from_wav(self.alarm_file)
                print(f"✅ Audio de alarma cargado: {self.alarm_file}")
            except Exception as e:
                print(f"⚠️  Error cargando audio: {e}")
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "alerts_sent": 0,
            "start_time": None,
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Deteniendo detección...")
        self.running = False
    
    def _play_alarm_async(self):
        """Play alarm sound in a separate thread."""
        if not AUDIO_AVAILABLE or self.alarm_sound is None:
            return
        
        current_time = time.time()
        if current_time - self.last_alarm_time < self.alarm_cooldown:
            return
        
        if not self.alarm_playing:
            self.alarm_playing = True
            self.last_alarm_time = current_time
            
            def _play():
                try:
                    play(self.alarm_sound)
                except Exception as e:
                    print(f"⚠️  Error reproduciendo alarma: {e}")
                finally:
                    self.alarm_playing = False
            
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
    
    def _print_metrics(self, metrics: dict):
        """Print metrics to console."""
        if not self.show_metrics:
            return
        
        # Clear line and print metrics
        ear = metrics.get("ear", 0)
        mar = metrics.get("mar", 0)
        perclos = metrics.get("perclos", 0)
        yawn_time = metrics.get("yawn_time", 0)
        head_tilt_time = metrics.get("head_tilt_time", 0)
        head_pose = metrics.get("head_pose", {})
        alert_text = metrics.get("alert_text")
        is_alarm = metrics.get("is_alarm", False)
        
        status = "🟢" if not is_alarm else "🔴"
        
        # Build status line
        line = (
            f"\r{status} EAR:{ear:.2f} | MAR:{mar:.2f} | "
            f"PERCLOS:{perclos:.1f}% | "
            f"Pitch:{head_pose.get('pitch', 0):.1f}° | "
            f"Yawn:{yawn_time:.1f}s | Tilt:{head_tilt_time:.1f}s"
        )
        
        if alert_text:
            line += f" | ⚠️  {alert_text}"
        
        # Print without newline, overwriting previous line
        print(line + " " * 10, end="", flush=True)
    
    def run(self):
        """Main detection loop."""
        print("=" * 60)
        print("🚗 DETECTOR DE SOMNOLENCIA - Modo Standalone")
        print("=" * 60)
        print(f"📌 Viaje ID: {self.viaje_id}")
        print(f"📷 Cámara: {self.camera_id}")
        print(f"🌐 API: {self.api_url}")
        print(f"🖥️  Modo: {'Headless (sin ventana)' if self.headless else 'Con ventana de video'}")
        print("=" * 60)
        print("Presiona 'q' para salir (si hay ventana) o Ctrl+C")
        print()
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir la cámara {self.camera_id}")
            print("💡 Verifica que la cámara esté conectada y no esté siendo usada por otra aplicación")
            return False
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Cámara abierta: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
        print()
        
        self.running = True
        self.stats["start_time"] = datetime.now()
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("\n⚠️  Error leyendo frame de la cámara")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame, play_alarm, metrics = self.video_handler.process(
                    frame, self.thresholds
                )
                
                self.stats["frames_processed"] += 1
                
                # Print metrics to console
                self._print_metrics(metrics)
                
                # Play alarm if needed
                if play_alarm:
                    self._play_alarm_async()
                
                # Display frame if not headless
                if not self.headless:
                    cv2.imshow("Detección de Somnolencia", processed_frame)
                    
                    # Check for 'q' key to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n\n👋 Saliendo por tecla 'q'...")
                        break
                    elif key == ord('r'):
                        # Reset PERCLOS
                        self.video_handler.reset_perclos()
                        print("\n🔄 PERCLOS reiniciado")
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"\n❌ Error durante la detección: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            
            # Print statistics
            self._print_final_stats()
        
        return True
    
    def _print_final_stats(self):
        """Print final statistics."""
        print("\n")
        print("=" * 60)
        print("📊 ESTADÍSTICAS FINALES")
        print("=" * 60)
        
        if self.stats["start_time"]:
            duration = datetime.now() - self.stats["start_time"]
            print(f"⏱️  Duración: {duration}")
        
        print(f"🎞️  Frames procesados: {self.stats['frames_processed']}")
        
        if self.stats["start_time"] and self.stats["frames_processed"] > 0:
            fps = self.stats["frames_processed"] / duration.total_seconds()
            print(f"📈 FPS promedio: {fps:.1f}")
        
        print("=" * 60)


def verify_trip_exists(api_url: str, viaje_id: int) -> bool:
    """Verify that the trip exists and is active."""
    import requests
    try:
        response = requests.get(f"{api_url}/viajes/{viaje_id}", timeout=5)
        if response.status_code == 200:
            trip = response.json()
            if trip.get("fecha_fin") is None:
                return True
            else:
                print(f"⚠️  El viaje {viaje_id} ya está finalizado")
                return False
        else:
            print(f"❌ Viaje {viaje_id} no encontrado")
            return False
    except requests.exceptions.ConnectionError:
        print(f"⚠️  No se pudo conectar al backend en {api_url}")
        print("   Continuando sin verificación...")
        return True
    except Exception as e:
        print(f"⚠️  Error verificando viaje: {e}")
        return True


def list_cameras() -> list:
    """List available camera devices."""
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Detector de somnolencia standalone (sin navegador)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python standalone_detector.py --viaje_id 123
  python standalone_detector.py --viaje_id 123 --camera 1
  python standalone_detector.py --viaje_id 123 --headless
  python standalone_detector.py --list-cameras
        """
    )
    
    parser.add_argument("--viaje_id", "-v", type=int, 
                        help="ID del viaje activo (requerido)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="ID de la cámara a usar (default: 0)")
    parser.add_argument("--headless", action="store_true",
                        help="Ejecutar sin ventana de video")
    parser.add_argument("--api_url", "-a", type=str, default="http://localhost:8000",
                        help="URL del backend API (default: http://localhost:8000)")
    parser.add_argument("--no-metrics", action="store_true",
                        help="No mostrar métricas en consola")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Listar cámaras disponibles y salir")
    
    args = parser.parse_args()
    
    # List cameras mode
    if args.list_cameras:
        print("🔍 Buscando cámaras disponibles...")
        cameras = list_cameras()
        if cameras:
            print(f"✅ Cámaras encontradas: {cameras}")
            for cam_id in cameras:
                cap = cv2.VideoCapture(cam_id)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"   Cámara {cam_id}: {w}x{h}")
        else:
            print("❌ No se encontraron cámaras")
        return
    
    # Validate viaje_id
    if args.viaje_id is None:
        print("❌ Error: Se requiere --viaje_id")
        print("   Usa --help para ver las opciones")
        sys.exit(1)
    
    # Verify trip exists
    print(f"🔍 Verificando viaje {args.viaje_id}...")
    if not verify_trip_exists(args.api_url, args.viaje_id):
        sys.exit(1)
    print(f"✅ Viaje {args.viaje_id} verificado")
    
    # Create and run detector
    detector = StandaloneDetector(
        viaje_id=args.viaje_id,
        camera_id=args.camera,
        headless=args.headless,
        api_url=args.api_url,
        show_metrics=not args.no_metrics,
    )
    
    success = detector.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
