#!/usr/bin/env python3
"""
Auto-start drowsiness detector for Raspberry Pi in vehicle.

This script runs automatically on boot using the SAME logic as streamlit_app.py
but without requiring a web interface. It detects the conductor, starts detection,
and sends data to the backend API automatically.
"""

import os
import sys
import time
import cv2
import signal
import argparse
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# Add project to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from drowsiness_detector import (
    VideoFrameHandler,
    AudioFrameHandler,
    get_api_client,
    DEFAULT_THRESHOLDS
)
from drowsiness_detector.audio_handler import play_alarm_sound


class AutoDetector:
    """Automatic drowsiness detector - same logic as Streamlit but without web UI."""

    def __init__(self, conductor_id=None, api_url=None, camera_index=0):
        """Initialize the auto detector.

        Args:
            conductor_id: ID of the conductor (if None, will fetch from API)
            api_url: Backend API URL
            camera_index: Camera device index (default: 0)
        """
        self.conductor_id = conductor_id
        self.camera_index = camera_index
        self.running = False
        self.viaje_id = None

        # Setup API client (same as Streamlit)
        self.api_client = get_api_client()
        if api_url:
            self.api_client.set_base_url(api_url)

        # Initialize video handler (will be set after getting viaje_id)
        self.video_handler = None

        # Audio setup (same as Streamlit)
        alarm_file = project_dir / "audio" / "wake_up.wav"
        if not alarm_file.exists():
            print(f"⚠️  Warning: Alarm file not found at {alarm_file}")
            self.audio_handler = None
        else:
            try:
                self.audio_handler = AudioFrameHandler(sound_file_path=str(alarm_file))
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize audio: {e}")
                self.audio_handler = None

        # Camera
        self.cap = None

        # Thresholds (same as Streamlit DEFAULT_THRESHOLDS)
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        self.thresholds["PERCLOS_THRESH"] = 15.0  # Default threshold

        # Shared state (same as Streamlit session_state)
        self.shared_state = {
            "play_alarm": False,
            "metrics": {
                "ear": 0.0,
                "mar": 0.0,
                "perclos": 0.0,
                "yawn_time": 0.0,
                "head_tilt_time": 0.0,
                "head_pose": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "alert_text": None,
                "tilt_type": None,
                "is_alarm": False,
            }
        }

        # Lock for thread-safe access to shared_state
        self.lock = threading.Lock()
        
        # Alarm state
        self.alarm_playing = False
        self.alarm_thread = None
        self.alarm_stop_event = threading.Event()  # Event to stop alarm loop
        
        # Detect if Raspberry Pi for optimization
        import platform
        self.is_raspberry = "arm" in platform.machine().lower() or "raspberry" in platform.uname().release.lower()
    
    def wait_for_api_connection(self, max_retries=60, retry_interval=5):
        """Wait for API connection to be established.

        Args:
            max_retries: Maximum number of connection attempts
            retry_interval: Seconds between attempts

        Returns:
            bool: True if connected, False otherwise
        """
        print(f"🔌 Esperando conexión con API: {self.api_client.base_url}")

        for attempt in range(1, max_retries + 1):
            if self.api_client.check_connection():
                print(f"✅ Conectado a la API en intento {attempt}")
                return True

            print(f"⏳ Intento {attempt}/{max_retries} - Reintentando en {retry_interval}s...")
            time.sleep(retry_interval)

        print(f"❌ No se pudo conectar a la API después de {max_retries} intentos")
        return False

    def get_or_wait_for_active_trip(self, check_interval=10):
        """Get active trip for conductor or wait for one to start.

        Args:
            check_interval: Seconds between checks

        Returns:
            dict: Trip data or None
        """
        print(f"🔍 Buscando viaje activo para conductor ID: {self.conductor_id}")

        while self.running:
            # Try to get active trip
            trip = self.api_client.get_active_trip_by_driver(self.conductor_id)

            if trip:
                self.viaje_id = trip.get("id_viaje")
                print(f"✅ Viaje activo encontrado: #{self.viaje_id}")
                return trip

            print(f"⏳ No hay viaje activo. Verificando nuevamente en {check_interval}s...")
            time.sleep(check_interval)

        return None

    def setup_camera(self):
        """Setup camera capture."""
        print(f"📷 Inicializando cámara (índice: {self.camera_index})...")

        self.cap = cv2.VideoCapture(self.camera_index)

        # Set camera properties for better performance on Raspberry Pi
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ No se pudo abrir la cámara {self.camera_index}")

        print("✅ Cámara inicializada")

    def setup_video_handler(self):
        """Setup video frame handler with trip ID - SAME as Streamlit."""
        print("🎥 Inicializando procesador de video...")

        # Create video handler only once (same as Streamlit optimization)
        if self.video_handler is None:
            self.video_handler = VideoFrameHandler(
                viaje_id=self.viaje_id,
                use_raspberry_pi_optimization=self.is_raspberry
            )
        else:
            # Update viaje_id without recreating handler (same as Streamlit fix)
            self.video_handler.update_viaje_id(self.viaje_id, reset_state=False)

        print("✅ Procesador de video inicializado")

    def process_frame(self, frame):
        """Process a single frame - EXACTLY the same as Streamlit video_frame_callback.

        Args:
            frame: Camera frame

        Returns:
            tuple: (processed_frame, should_alarm, metrics)
        """
        if self.video_handler is None:
            return frame, False, self.shared_state["metrics"]

        try:
            # Process frame with error handling (same as Streamlit)
            processed_frame, play_alarm, metrics = self.video_handler.process(
                frame, self.thresholds
            )

            # Validate processed frame (same as Streamlit)
            if processed_frame is None or processed_frame.size == 0:
                processed_frame = frame
                play_alarm = False
                metrics = self.shared_state["metrics"]

            # Update shared state (same as Streamlit)
            with self.lock:
                self.shared_state["play_alarm"] = play_alarm
                self.shared_state["metrics"] = metrics

            return processed_frame, play_alarm, metrics

        except Exception as e:
            # If processing fails, return original frame (same as Streamlit)
            import traceback
            print(f"⚠️  Error processing frame: {e}")
            print(traceback.format_exc())
            return frame, False, self.shared_state["metrics"]

    def handle_alarm(self, should_alarm):
        """Handle alarm state changes - plays sound CONTINUOUSLY while alarm is active.

        Args:
            should_alarm: Whether alarm should be playing
        """
        # Alarm started - create continuous alarm loop
        if should_alarm and not self.alarm_playing:
            print("🚨 ALERTA DE SOMNOLENCIA - Activando alarma continua")
            self.alarm_playing = True
            self.alarm_stop_event.clear()

            # Play alarm sound continuously in separate thread
            if self.audio_handler:
                def play_alarm_loop():
                    """Play alarm sound continuously until stopped."""
                    alarm_path = str(project_dir / "audio" / "wake_up.wav")
                    print("🔊 Iniciando reproducción continua de alarma...")
                    
                    while not self.alarm_stop_event.is_set():
                        try:
                            # Play alarm sound (duration ~2 seconds)
                            play_alarm_sound(alarm_path, duration=2.0)
                            
                            # Small pause before repeating (0.5 seconds)
                            # Total cycle: ~2.5 seconds per loop
                            if not self.alarm_stop_event.wait(0.5):
                                continue
                            else:
                                break
                                
                        except Exception as e:
                            print(f"⚠️  Error reproduciendo alarma: {e}")
                            # Wait before retrying
                            if self.alarm_stop_event.wait(1.0):
                                break
                    
                    print("🔇 Alarma detenida")

                # Start alarm thread
                self.alarm_thread = threading.Thread(target=play_alarm_loop, daemon=True)
                self.alarm_thread.start()

        # Alarm stopped - signal thread to stop
        elif not should_alarm and self.alarm_playing:
            print("✅ Estado normal - Desactivando alarma")
            self.alarm_playing = False
            self.alarm_stop_event.set()  # Signal the alarm thread to stop
            
            # Wait briefly for thread to finish
            if self.alarm_thread and self.alarm_thread.is_alive():
                self.alarm_thread.join(timeout=1.0)

    def run(self):
        """Main detection loop."""
        self.running = True

        print("=" * 60)
        print("🚗 DETECTOR DE SOMNOLENCIA - MODO AUTOMÁTICO")
        print("=" * 60)

        try:
            # Step 1: Wait for API connection
            if not self.wait_for_api_connection():
                print("❌ No se puede ejecutar sin conexión a la API")
                return

            # Step 2: Get conductor ID if not provided
            if self.conductor_id is None:
                print("⚠️  No se especificó ID de conductor")
                # Try to get first available driver
                drivers = self.api_client.get_drivers()
                if drivers:
                    self.conductor_id = drivers[0].get("id_conductor")
                    print(f"📋 Usando conductor: {drivers[0].get('nombre')} (ID: {self.conductor_id})")
                else:
                    print("❌ No hay conductores disponibles en el sistema")
                    return

            # Step 3: Wait for active trip
            trip = self.get_or_wait_for_active_trip()
            if not trip:
                print("❌ No se pudo obtener viaje activo")
                return

            # Step 4: Setup camera
            self.setup_camera()

            # Step 5: Setup video handler
            self.setup_video_handler()

            # Step 6: Main detection loop
            print("\n" + "=" * 60)
            print("🎬 INICIANDO DETECCIÓN")
            print("=" * 60)
            print(f"📍 Viaje: #{self.viaje_id}")
            print(f"👤 Conductor: ID {self.conductor_id}")
            print(f"🔗 API: {self.api_client.base_url}")
            print(f"⚙️  Umbral PERCLOS: {self.thresholds['PERCLOS_THRESH']}%")
            print("=" * 60 + "\n")

            frame_count = 0
            start_time = time.time()
            last_status_time = start_time

            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  No se pudo leer frame de la cámara")
                    time.sleep(0.1)
                    continue

                # Process frame (SAME logic as Streamlit video_frame_callback)
                processed_frame, should_alarm, metrics = self.process_frame(frame)

                # Handle alarm (SAME logic as Streamlit audio_frame_callback)
                self.handle_alarm(should_alarm)

                # Update frame counter
                frame_count += 1

                # Print status every 5 seconds (more details like Streamlit displays)
                current_time = time.time()
                if current_time - last_status_time >= 5.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed

                    # Get metrics from shared state (same as Streamlit)
                    with self.lock:
                        metrics_copy = self.shared_state["metrics"].copy()
                        is_alarm = self.shared_state["play_alarm"]

                    # Format status (similar to Streamlit display)
                    status = "🚨 ALARMA ACTIVA" if is_alarm else "✅ Normal"

                    print(f"\n📊 Estado [{datetime.now().strftime('%H:%M:%S')}]:")
                    print(f"   {'='*50}")
                    print(f"   Status: {status}")
                    print(f"   {'='*50}")
                    print(f"   • EAR (Eye):        {metrics_copy.get('ear', 0):.3f}  {'(CERRADO)' if metrics_copy.get('ear', 0) < 0.18 else '(ABIERTO)'}")
                    print(f"   • MAR (Mouth):      {metrics_copy.get('mar', 0):.3f}  {'(BOSTEZO)' if metrics_copy.get('mar', 0) > 0.6 else '(NORMAL)'}")
                    print(f"   • PERCLOS:          {metrics_copy.get('perclos', 0):.1f}%  {'(PELIGRO)' if metrics_copy.get('perclos', 0) > 15 else '(OK)'}")
                    print(f"   • Tiempo Bostezo:   {metrics_copy.get('yawn_time', 0):.1f}s")
                    print(f"   • Tiempo Cabeceo:   {metrics_copy.get('head_tilt_time', 0):.1f}s")

                    # Head pose details
                    head_pose = metrics_copy.get('head_pose', {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
                    print(f"   • Head Pose:")
                    print(f"     - Roll (lateral):  {head_pose['roll']:.1f}°")
                    print(f"     - Pitch (arriba/abajo): {head_pose['pitch']:.1f}°")
                    print(f"     - Yaw (izq/der):   {head_pose['yaw']:.1f}°")

                    # Alert text
                    alert_text = metrics_copy.get('alert_text')
                    if alert_text:
                        print(f"   • Alerta: {alert_text}")

                    print(f"   {'='*50}")
                    print(f"   • Frames: {frame_count} | FPS: {fps:.1f}")
                    print(f"   • Viaje ID: {self.viaje_id}")
                    print()

                    last_status_time = current_time

                # Small delay to prevent 100% CPU usage (same as frame rate control)
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n⏹️  Deteniendo detector...")

        except Exception as e:
            print(f"\n❌ Error fatal: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 Limpiando recursos...")

        self.running = False
        
        # Stop alarm if playing
        if self.alarm_playing:
            print("⏹️  Deteniendo alarma...")
            self.alarm_stop_event.set()
            if self.alarm_thread and self.alarm_thread.is_alive():
                self.alarm_thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()
            print("✅ Cámara liberada")

        cv2.destroyAllWindows()

        print("✅ Limpieza completada")

    def signal_handler(self, signum, frame):
        """Handle system signals."""
        print(f"\n⚠️  Señal recibida: {signum}")
        self.running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detector automático de somnolencia para vehículos"
    )
    parser.add_argument(
        "--conductor-id",
        type=int,
        default=None,
        help="ID del conductor (opcional, se detectará automáticamente)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("RISK_ADVISOR_API_URL", "http://192.168.100.82:8000"),
        help="URL del backend API"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de la cámara (default: 0)"
    )
    parser.add_argument(
        "--perclos-threshold",
        type=float,
        default=30.0,
        help="Umbral PERCLOS en porcentaje (default: 15.0)"
    )

    args = parser.parse_args()

    # Create detector
    detector = AutoDetector(
        conductor_id=args.conductor_id,
        api_url=args.api_url,
        camera_index=args.camera
    )

    # Set threshold
    detector.thresholds["PERCLOS_THRESH"] = args.perclos_threshold

    # Setup signal handlers
    signal.signal(signal.SIGINT, detector.signal_handler)
    signal.signal(signal.SIGTERM, detector.signal_handler)

    # Run detector
    detector.run()


if __name__ == "__main__":
    main()
