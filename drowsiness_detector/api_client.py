"""
API client for communicating with the risk-advisor backend.

Handles sending detection events, readings, and alerts to the backend API.
"""

import os
import threading
import requests
from typing import Optional, Dict, Any
from datetime import datetime


API_BASE_URL = os.getenv("RISK_ADVISOR_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 5  # seconds


class APIClient:
    """Client for sending data to the risk-advisor backend API."""

    def __init__(self, base_url: str = None):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API (defaults to RISK_ADVISOR_API_URL env var)
        """
        self.base_url = base_url or API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def set_base_url(self, base_url: str):
        """Set the base URL for the API client.

        Args:
            base_url: New base URL for the API
        """
        self.base_url = base_url

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint (without base URL)
            data: Request body data

        Returns:
            Response JSON data or None if request fails
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json() if response.content else None
        except requests.exceptions.ConnectionError:
            # Connection errors are handled silently - connection status is checked separately
            return None
        except requests.exceptions.Timeout:
            # Timeout errors are handled silently
            return None
        except requests.exceptions.RequestException as e:
            # Only log non-connection errors
            print(f"API request failed: {e}")
            return None

    def get_active_trip_by_driver(self, conductor_id: int) -> Optional[Dict]:
        """Get the active trip for a driver.

        Args:
            conductor_id: ID of the driver

        Returns:
            Trip data or None if no active trip
        """
        url = f"{self.base_url}/viajes/conductor/{conductor_id}/activo"
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return None  # No active trip or conductor not found
            response.raise_for_status()
            return response.json() if response.content else None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Connection errors are handled silently - connection status is checked separately
            return None
        except requests.exceptions.RequestException:
            # Other request errors are handled silently
            return None

    def get_all_active_trips(self) -> list:
        """Get all active trips in the system.

        Returns:
            List of active trip data or empty list if none found
        """
        url = f"{self.base_url}/viajes/activos"
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json() if response.content else []
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
            # Connection errors are handled silently
            return []

    def get_drivers(self) -> list:
        """Get all drivers from the system.

        Returns:
            List of driver data or empty list if none found
        """
        url = f"{self.base_url}/conductores/"
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json() if response.content else []
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
            # Connection errors are handled silently
            return []

    def check_connection(self) -> bool:
        """Check if the API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        url = f"{self.base_url}/health"
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
            # Connection errors are expected when server is not running
            # Don't print errors here - let the UI handle the messaging
            return False

    def create_reading(self, reading_data: Dict[str, Any]) -> Optional[Dict]:
        """Create a sensor reading.

        Args:
            reading_data: Reading data including id_viaje, EAR, MAR, etc.

        Returns:
            Created reading data or None if request fails
        """
        return self._make_request("POST", "/lecturas/", reading_data)

    def create_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict]:
        """Create an alert.

        Args:
            alert_data: Alert data including id_viaje, tipo_alerta, etc.

        Returns:
            Created alert data or None if request fails
        """
        return self._make_request("POST", "/alertas/", alert_data)


# Global API client instance
_api_client = None


def get_api_client() -> APIClient:
    """Get or create the global API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client


def send_reading_async(reading_data: Dict[str, Any]):
    """Send a reading to the API asynchronously.

    Args:
        reading_data: Reading data to send
    """
    def _send():
        client = get_api_client()
        client.create_reading(reading_data)

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()


def send_alert_async(alert_data: Dict[str, Any]):
    """Send an alert to the API asynchronously.

    Args:
        alert_data: Alert data to send
    """
    def _send():
        client = get_api_client()
        client.create_alert(alert_data)

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()

