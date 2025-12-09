# Fix: Preventing VideoFrameHandler Instance Recreation

## Problem

The `streamlit_app.py` was creating a new `VideoFrameHandler` instance every time the `viaje_id` changed, and creating a new `AudioFrameHandler` instance on every app rerun. This caused several issues:

1. **Performance degradation**: Each time a new instance was created, MediaPipe FaceMesh was reinitialized, which is computationally expensive
2. **Memory waste**: Old instances weren't being properly cleaned up, leading to memory leaks
3. **State loss**: Detection state (PERCLOS tracking, alarm states, etc.) was reset unnecessarily
4. **Resource waste**: Audio files were reloaded unnecessarily on every rerun

### Root Cause

```python
# OLD CODE - Creates new instance when viaje_id changes
if "video_handler" not in st.session_state or st.session_state.get("last_viaje_id") != viaje_id:
    st.session_state["video_handler"] = VideoFrameHandler(
        viaje_id=viaje_id,
        use_raspberry_pi_optimization=is_raspberry
    )
```

## Solution

The fix involves three changes:

### 1. Modified `streamlit_app.py` - VideoFrameHandler

-  Create `VideoFrameHandler` **only once** when first needed
-  Update the `viaje_id` property directly instead of recreating the instance
-  This preserves the MediaPipe FaceMesh instance and all internal state

```python
# NEW CODE - Create once, update viaje_id as needed
if "video_handler" not in st.session_state:
    st.session_state["video_handler"] = VideoFrameHandler(
        viaje_id=viaje_id,
        use_raspberry_pi_optimization=is_raspberry
    )
    st.session_state["last_viaje_id"] = viaje_id

# Update viaje_id if it changed (without recreating the handler)
if st.session_state.get("last_viaje_id") != viaje_id:
    video_handler.update_viaje_id(viaje_id, reset_state=False)
    st.session_state["last_viaje_id"] = viaje_id
```

### 2. Modified `streamlit_app.py` - AudioFrameHandler

-  Cache `AudioFrameHandler` in session state
-  Prevents reloading the audio file on every app rerun

```python
# NEW CODE - Cache audio handler
if "audio_handler" not in st.session_state:
    try:
        st.session_state["audio_handler"] = AudioFrameHandler(sound_file_path=alarm_file_path)
    except Exception as e:
        st.error(f"Error al inicializar el procesador de audio: {e}")
        st.stop()

audio_handler = st.session_state.get("audio_handler")
```

### 3. Added `update_viaje_id()` method to `VideoFrameHandler`

Added a new method in `detection.py` to properly update the trip ID:

```python
def update_viaje_id(self, viaje_id: int, reset_state: bool = False):
    """Update the viaje_id for this handler.

    Args:
        viaje_id: New trip ID to associate readings with
        reset_state: If True, also reset all detection state
    """
    self.viaje_id = viaje_id
    if reset_state:
        self.reset_perclos()
    # Reset reading timer to send reading immediately with new trip
    self.last_reading_time = 0.0
```

## Benefits

-  **Better performance**: MediaPipe FaceMesh is initialized only once
-  **Lower memory usage**: No unnecessary instance recreation
-  **Consistent state**: Detection state is preserved across trip changes (unless explicitly reset)
-  **Immediate readings**: First reading is sent immediately when switching trips
-  **Faster app reruns**: Audio file is loaded once and cached

## Testing

After applying this fix:

1. Start the streamlit app
2. Connect to a trip
3. Switch to a different trip
4. Verify that the app continues working without lag or memory issues
5. Check that readings are sent to the new trip immediately

## Notes

-  The `reset_state` parameter in `update_viaje_id()` allows explicit state reset if needed
-  The fix maintains backward compatibility with existing code
-  MediaPipe instance lifecycle is now properly managed
