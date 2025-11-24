"""
Streamlit application for driver drowsiness detection.

This is the main entry point for the drowsiness detection web application.
"""

import os
import av
import cv2
import time
import threading
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

from drowsiness_detector import VideoFrameHandler, AudioFrameHandler, DEFAULT_THRESHOLDS, get_api_client


# Define the audio file to use.
alarm_file_path = os.path.join("audio", "wake_up.wav")

# Verify audio file exists
if not os.path.exists(alarm_file_path):
    st.error(f"Audio file not found: {alarm_file_path}")
    st.stop()

# Streamlit Components web page config
st.set_page_config(
    page_title="Detección de Somnolencia",
    page_icon="",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
    menu_items={
    },
)

# API Configuration - Allow user to set URL in session state
if "api_url" not in st.session_state:
    # Try to get from environment variable first, then default
    st.session_state["api_url"] = os.getenv("RISK_ADVISOR_API_URL", "http://192.168.100.82:8000")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuración")

    # API URL Configuration
    st.subheader("Configuración de API")
    api_url_input = st.text_input(
        "URL del Backend API",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        help="Ingrese la URL completa del backend (ej: http://192.168.1.100:8000 o http://localhost:8000)",
        key="api_url_input"
    )

    # Camera Configuration (for Raspberry Pi)
    st.subheader("Configuración de Cámara")

    # Camera mode selection
    camera_mode = st.radio(
        "Modo de cámara",
        ["OpenCV (Recomendado para SSH)", "WebRTC (Navegador)"],
        help="OpenCV accede directamente a la cámara del servidor. WebRTC usa la cámara del navegador.",
        index=0,  # Default to OpenCV
        key="camera_mode_radio"
    )

    # Save camera mode to session state
    st.session_state["camera_mode"] = camera_mode

    # Camera device selection for OpenCV
    if camera_mode == "OpenCV (Recomendado para SSH)":
        camera_index = st.number_input(
            "Índice de cámara",
            min_value=0,
            max_value=10,
            value=st.session_state.get("camera_index", 0),
            help="Índice de la cámara (0 para /dev/video0, 1 para /dev/video1, etc.)",
            key="camera_index_input"
        )
        st.session_state["camera_index"] = camera_index

        # Auto-refresh toggle (allows user to disable auto-refresh for scrolling)
        auto_refresh = st.checkbox(
            "Auto-refresh (actualización automática)",
            value=st.session_state.get("auto_refresh", True),
            help="Desactiva esto si quieres poder hacer scroll en la página. Usa el botón 'Actualizar Frame' para actualizar manualmente.",
            key="auto_refresh_checkbox"
        )
        st.session_state["auto_refresh"] = auto_refresh

    camera_info = st.info("💡 Si tienes problemas con la cámara, verifica los permisos y la conexión")

    # Check if running on Raspberry Pi
    import platform
    is_raspberry = "arm" in platform.machine().lower() or "raspberry" in platform.uname().release.lower()

    if is_raspberry:
        st.warning("⚠️ Detectado: Raspberry Pi")
        st.info("En Raspberry Pi, asegúrate de que la cámara esté habilitada:")
        st.code("sudo raspi-config  # Enable camera interface", language="bash")

    # Update API client when URL changes
    if api_url_input != st.session_state.get("api_url"):
        st.session_state["api_url"] = api_url_input
        api_client = get_api_client()
        api_client.base_url = api_url_input
        # Clear connection status to force recheck
        if "api_connected" in st.session_state:
            del st.session_state["api_connected"]

    # Initialize API client with current URL
    api_client = get_api_client()
    api_client.base_url = st.session_state["api_url"]

    # API Connection Status
    st.subheader("Estado de Conexión")

    # Check connection (cache result to avoid too many requests)
    if "api_connected" not in st.session_state:
        with st.spinner("Verificando conexión..."):
            st.session_state["api_connected"] = api_client.check_connection()

    api_connected = st.session_state["api_connected"]

    if api_connected:
        st.success(f"✅ Conectado a la API")
        st.caption(f"URL: {st.session_state['api_url']}")
    else:
        st.error(f"❌ No se puede conectar a la API")
        st.caption(f"URL intentada: {st.session_state['api_url']}")
        st.info("💡 Verifica que:")
        st.info("• El servidor backend esté ejecutándose")
        st.info("• La URL sea correcta (ej: http://IP:8000)")
        st.info("• No haya firewall bloqueando la conexión")

        if st.button("🔄 Reintentar Conexión", use_container_width=False):
            st.session_state["api_connected"] = None
            st.rerun()

    # Driver selection
    st.subheader("Seleccionar Conductor")

    # Try to load drivers from API
    drivers = []
    if api_connected:
        try:
            drivers = api_client.get_drivers()
        except Exception as e:
            st.warning(f"No se pudieron cargar los conductores: {str(e)}")

    if drivers:
        # Show dropdown with driver names
        driver_options = {f"{d.get('nombre', 'Sin nombre')} (ID: {d.get('id_conductor')})": d.get('id_conductor')
                         for d in drivers}
        selected_driver = st.selectbox(
            "Conductor",
            options=list(driver_options.keys()),
            index=0,
            help="Seleccione el conductor de la lista"
        )
        conductor_id = driver_options[selected_driver]
    else:
        # Fallback to number input
        conductor_id = st.number_input(
            "ID del Conductor",
            min_value=1,
            value=st.session_state.get("conductor_id", 1),
            step=1,
            help="Ingrese el ID del conductor para el viaje"
        )

    # Get active trip
    viaje_id = None

    # Auto-detect active trip on page load if not already connected
    # Only try if API is connected
    if api_connected and not st.session_state.get("viaje_id") and not st.session_state.get("auto_detect_attempted"):
        with st.spinner("Buscando viaje activo..."):
            try:
                active_trip = api_client.get_active_trip_by_driver(conductor_id)
                if active_trip:
                    viaje_id = active_trip.get("id_viaje")
                    st.session_state["viaje_id"] = viaje_id
                    st.session_state["conductor_id"] = conductor_id
                    st.success(f"✅ Viaje activo encontrado: #{viaje_id}")
                st.session_state["auto_detect_attempted"] = True
            except Exception as e:
                st.session_state["auto_detect_attempted"] = True
                pass  # Silently fail on auto-detect

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Buscar Viaje Activo", use_container_width=False):
            if api_connected:
                try:
                    with st.spinner("Buscando viaje activo..."):
                        active_trip = api_client.get_active_trip_by_driver(conductor_id)
                        if active_trip:
                            viaje_id = active_trip.get("id_viaje")
                            st.session_state["viaje_id"] = viaje_id
                            st.session_state["conductor_id"] = conductor_id
                            st.session_state["auto_detect_attempted"] = True
                            st.success(f"✅ Conectado al viaje #{viaje_id}")
                            st.rerun()
                        else:
                            st.warning("⚠️ No hay un viaje activo para este conductor.")
                            st.info("💡 El administrador debe iniciar un viaje primero desde el panel de administración.")
                            st.session_state["viaje_id"] = None
                            st.session_state["auto_detect_attempted"] = True
                except Exception as e:
                    st.error(f"❌ Error al conectar: {str(e)}")
                    st.session_state["viaje_id"] = None
            else:
                st.error("❌ No hay conexión con la API")
                st.info("💡 Configura la URL del backend y verifica la conexión antes de buscar viajes.")

    with col2:
        if st.button("📋 Ver Todos los Activos", use_container_width=False):
            if api_connected:
                try:
                    active_trips = api_client.get_all_active_trips()
                    if active_trips:
                        st.info(f"📊 Se encontraron {len(active_trips)} viaje(s) activo(s):")
                        for trip in active_trips:
                            trip_id = trip.get("id_viaje")
                            trip_conductor = trip.get("id_conductor")
                            st.text(f"  • Viaje #{trip_id} - Conductor ID: {trip_conductor}")
                    else:
                        st.warning("⚠️ No hay viajes activos en el sistema")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ No hay conexión con la API")

    # Show current connection status
    st.divider()
    if st.session_state.get("viaje_id"):
        st.success(f"🔗 Conectado al viaje #{st.session_state['viaje_id']}")
        st.info(f"👤 Conductor ID: {st.session_state.get('conductor_id', 'N/A')}")
        if st.button("🔌 Desconectar", use_container_width=False):
            st.session_state["viaje_id"] = None
            st.session_state["last_viaje_id"] = None
            st.session_state["auto_detect_attempted"] = False
            st.rerun()
    else:
        st.warning("⚠️ No conectado a ningún viaje")
        st.info("💡 Use 'Buscar Viaje Activo' para conectarse automáticamente")

# Use default thresholds from config
thresholds = DEFAULT_THRESHOLDS.copy()

# Get viaje_id from session state
viaje_id = st.session_state.get("viaje_id")

# Initialize handlers (video_handler will be recreated if viaje_id changes)
if "video_handler" not in st.session_state or st.session_state.get("last_viaje_id") != viaje_id:
    st.session_state["video_handler"] = VideoFrameHandler(viaje_id=viaje_id)
    st.session_state["last_viaje_id"] = viaje_id

# Get handlers with error checking
video_handler = st.session_state.get("video_handler")
if video_handler is None:
    st.error("⚠️ El procesador de video no está inicializado")
    st.stop()

try:
    audio_handler = AudioFrameHandler(sound_file_path=alarm_file_path)
except Exception as e:
    st.error(f"Error al inicializar el procesador de audio: {e}")
    st.stop()

lock = threading.Lock()  # For thread-safe access & to prevent race-condition.
shared_state = {"play_alarm": False}


def video_frame_callback(frame: av.VideoFrame):
    """Callback function to process video frames."""
    try:
        # Convert frame to numpy array
        frame_array = frame.to_ndarray(format="bgr24")

        # Validate frame
        if frame_array is None or frame_array.size == 0:
            return frame

        # Process frame with error handling and timeout protection
        try:
            processed_frame, play_alarm = video_handler.process(frame_array, thresholds)

            # Validate processed frame
            if processed_frame is None or processed_frame.size == 0:
                processed_frame = frame_array
                play_alarm = False

        except Exception as e:
            # If processing fails, return original frame
            import traceback
            print(f"Error processing frame: {e}")
            print(traceback.format_exc())
            processed_frame = frame_array
            play_alarm = False

        # Update shared state
        with lock:
            shared_state["play_alarm"] = play_alarm

        # Return processed frame
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
    except Exception as e:
        # If callback fails completely, return original frame
        import traceback
        print(f"Error in video_frame_callback: {e}")
        print(traceback.format_exc())
        return frame


def audio_frame_callback(frame: av.AudioFrame):
    try:
        with lock:  # access the current "play_alarm" state
            play_alarm = shared_state.get("play_alarm", False)

        new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_alarm)
        return new_frame
    except Exception as e:
        # If audio callback fails, return original frame
        print(f"Error in audio_frame_callback: {e}")
        return frame

st.header("📹 Detección de Somnolencia")

# Check if we have a trip connection
if not st.session_state.get("viaje_id"):
    st.warning("⚠️ Debes conectarte a un viaje activo antes de iniciar la detección")
    st.info("💡 Usa el botón 'Buscar Viaje Activo' en la barra lateral")
else:
    st.info(f"✅ Conectado al viaje #{st.session_state.get('viaje_id')} - La detección está activa")

    # Get camera mode from session state (set in sidebar)
    camera_mode = st.session_state.get("camera_mode", "OpenCV (Recomendado para SSH)")
    camera_index = st.session_state.get("camera_index", 0)

    if camera_mode == "OpenCV (Recomendado para SSH)":
        # OpenCV mode - Direct camera access from server
        st.subheader("Modo OpenCV - Acceso directo a la cámara del servidor")

        # Initialize camera capture in session state
        if "camera_capture" not in st.session_state:
            st.session_state["camera_capture"] = None
            st.session_state["detection_running"] = False

        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Iniciar Detección", disabled=st.session_state.get("detection_running", False)):
                try:
                    # Try to open camera
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        st.error(f"❌ No se pudo abrir la cámara {camera_index}")
                        st.info("💡 Verifica que la cámara esté conectada y que tengas permisos")
                    else:
                        st.session_state["camera_capture"] = cap
                        st.session_state["detection_running"] = True
                        st.session_state["frame_count"] = 0
                        st.success("✅ Cámara iniciada correctamente")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al abrir la cámara: {e}")

        with col2:
            if st.button("⏹️ Detener Detección", disabled=not st.session_state.get("detection_running", False)):
                if st.session_state.get("camera_capture") is not None:
                    st.session_state["camera_capture"].release()
                    st.session_state["camera_capture"] = None
                st.session_state["detection_running"] = False
                st.session_state["frame_count"] = 0
                st.info("⏸️ Detección detenida")
                st.rerun()

        with col3:
            # Manual refresh button (allows user to control updates)
            if st.button("🔄 Actualizar Frame", disabled=not st.session_state.get("detection_running", False)):
                st.rerun()

        # Video display and processing
        if st.session_state.get("detection_running", False) and st.session_state.get("camera_capture") is not None:
            cap = st.session_state["camera_capture"]

            # Create placeholder for video
            video_placeholder = st.empty()

            # Status indicator
            status_placeholder = st.empty()

            # Process single frame per execution (allows user interaction)
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    status_placeholder.error("❌ No se pudo leer el frame de la cámara")
                    # Stop detection if camera fails
                    cap.release()
                    st.session_state["camera_capture"] = None
                    st.session_state["detection_running"] = False
                else:
                    # Process frame
                    try:
                        processed_frame, play_alarm = video_handler.process(frame, thresholds)

                        # Update shared state for audio
                        with lock:
                            shared_state["play_alarm"] = play_alarm

                        # Convert BGR to RGB for display
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                        # Display frame
                        video_placeholder.image(display_frame, channels="RGB", width=None)

                        # Update status (only show occasionally to reduce updates)
                        if "frame_count" not in st.session_state:
                            st.session_state["frame_count"] = 0
                        st.session_state["frame_count"] += 1

                        if st.session_state["frame_count"] % 30 == 0:  # Update status every 30 frames
                            status_placeholder.info(f"✅ Detección activa - Frame {st.session_state['frame_count']}")

                        # Play alarm if needed (simple beep for now)
                        if play_alarm:
                            # Audio will be handled separately if needed
                            pass

                    except Exception as e:
                        import traceback
                        status_placeholder.error(f"Error procesando frame: {e}")
                        print(traceback.format_exc())
                        # Show original frame on error
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(display_frame, channels="RGB", width=None)

                    # Auto-refresh option (can be disabled to allow scrolling)
                    auto_refresh = st.session_state.get("auto_refresh", True)

                    # Rerun only if detection is still running and auto-refresh is enabled
                    # This allows user to scroll when auto-refresh is disabled
                    if st.session_state.get("detection_running", False) and auto_refresh:
                        time.sleep(0.03)  # ~30 FPS
                        st.rerun()
                    elif st.session_state.get("detection_running", False) and not auto_refresh:
                        # Show message that user can manually refresh (only once)
                        if "manual_refresh_shown" not in st.session_state:
                            status_placeholder.info("💡 Auto-refresh desactivado. Usa el botón 'Actualizar Frame' para ver el siguiente frame, o activa 'Auto-refresh' para actualización automática")
                            st.session_state["manual_refresh_shown"] = True

            except Exception as e:
                status_placeholder.error(f"❌ Error en el loop de detección: {e}")
                import traceback
                print(traceback.format_exc())
                # Cleanup on error
                if st.session_state.get("camera_capture") is not None:
                    st.session_state["camera_capture"].release()
                    st.session_state["camera_capture"] = None
                st.session_state["detection_running"] = False
        else:
            st.info("💡 Haz clic en 'Iniciar Detección' para comenzar")

    else:
        # WebRTC mode - Browser camera access
        st.subheader("Modo WebRTC - Cámara del navegador")

        with st.expander("ℹ️ Instrucciones para acceder a la cámara"):
            st.markdown("""
            **Para usar la cámara en tu navegador:**

            1. **Permisos del navegador**: Cuando se cargue el video, el navegador te pedirá permiso para acceder a la cámara. Debes aceptar.

            2. **En Raspberry Pi**:
               - Si accedes desde otra máquina: El navegador usará la cámara de esa máquina
               - Si accedes localmente: El navegador usará la cámara de la Raspberry Pi (si está disponible)

            3. **Verificar cámara disponible**:
               ```bash
               # En Raspberry Pi, verifica dispositivos de video:
               v4l2-ctl --list-devices
               ls -l /dev/video*
               ```

            4. **Si el error persiste**:
               - Verifica que la cámara esté conectada y funcionando
               - Prueba con otro navegador (Chrome, Firefox)
               - Verifica que la URL sea `http://` o `https://` (no `file://`)
               - Asegúrate de que el navegador tenga permisos de cámara
            """)

        st.info("💡 **Instrucciones**: Haz clic en el botón 'Start' debajo para iniciar la detección")
        st.info("⚠️ **Importante**: Acepta el permiso de cámara cuando el navegador lo solicite")

        try:
            ctx = webrtc_streamer(
                key="drowsiness-detection",
                video_frame_callback=video_frame_callback,
                audio_frame_callback=audio_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={
                    "video": {
                        "height": {"ideal": 480},
                        "width": {"ideal": 640},
                        "facingMode": "user"  # Front-facing camera
                    },
                    "audio": True
                },
                video_html_attrs=VideoHTMLAttributes(
                    autoPlay=True,
                    controls=False,
                    muted=False,
                    style={"width": "100%"}
                ),
            )

            # Check state after initialization
            if ctx:
                if hasattr(ctx, 'state'):
                    if ctx.state.playing:
                        st.success("✅ Cámara activa - Detección en curso")
                    elif ctx.state.playing is False:
                        st.info("⏸️ Cámara pausada - Haz clic en 'Start' para iniciar")
                    else:
                        st.warning("⚠️ Esperando acceso a la cámara...")
                        st.info("💡 Acepta el permiso de cámara cuando el navegador lo solicite")
                else:
                    st.info("💡 Haz clic en 'Start' para iniciar la detección")
            else:
                st.warning("⚠️ Componente de video no inicializado")

        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error al acceder a la cámara: {error_msg}")

            if "NotFoundError" in error_msg or "device not found" in error_msg.lower():
                st.error("**Problema**: No se encontró ningún dispositivo de cámara")
                st.markdown("""
                **Soluciones:**

                1. **Verifica que la cámara esté conectada**:
                   - En Raspberry Pi: `lsusb` o `v4l2-ctl --list-devices`
                   - Verifica que aparezca `/dev/video0` o similar

                2. **Permisos del navegador**:
                   - Asegúrate de que el navegador tenga permisos para acceder a la cámara
                   - Verifica la configuración de privacidad del navegador

                3. **Acceso HTTPS/HTTP**:
                   - Algunos navegadores requieren HTTPS para acceder a la cámara
                   - Prueba accediendo con `https://` si es posible
                   - O usa `http://localhost` en lugar de la IP

                4. **Reinicia la aplicación**:
                   ```bash
                   # Detén Streamlit (Ctrl+C) y reinicia:
                   streamlit run streamlit_app.py
                   ```
                """)
            else:
                st.error(f"Error desconocido: {error_msg}")
                st.info("💡 Intenta recargar la página o reiniciar la aplicación")

