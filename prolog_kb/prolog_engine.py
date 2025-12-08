"""
Prolog engine bridge:
---------------------
 - Carga la base de conocimiento declarativa (base_conocimiento.pl)
 - Obtiene datos reales desde PostgreSQL
 - Inserta hechos dinámicos en Prolog usando PySWIP
 - Expone funciones para evaluar reglas (somnolencia crítica, acciones, etc.)

Este módulo puede ser importado tanto por el backend FastAPI como por scripts de
diagnóstico (ver test_prolog_engine.py).
"""

from __future__ import annotations

import os
from typing import Dict, List

from pyswip import Prolog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ruta relativa al archivo Prolog
PROLOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "base_conocimiento.pl",
)

# SQLAlchemy session (usa la misma URL que el backend)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/risk_advisor")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class PrologSomnolenciaEngine:
    def __init__(self) -> None:
        self.prolog = Prolog()
        self.prolog.consult(PROLOG_FILE)

    # ------------------------------------------------------------------ util
    def _reset_dynamic_facts(self) -> None:
        """Elimina hechos dinámicos previos antes de volver a cargarlos."""
        self.prolog.retractall("conductor(_,_,_,_)")
        self.prolog.retractall("viaje(_,_,_,_)")
        self.prolog.retractall("lectura_somnolencia(_,_,_)")
        self.prolog.retractall("alerta_emitida(_,_,_)")

    def _sanitize(self, value: str) -> str:
        return value.replace("'", "\\'")

    # ------------------------------------------------------------- carga DB
    def cargar_desde_db(self) -> None:
        """Lee tablas relevantes y las inserta como hechos Prolog."""
        self._reset_dynamic_facts()
        db = SessionLocal()
        try:
            self._cargar_conductores(db)
            self._cargar_viajes(db)
            self._cargar_lecturas(db)
            self._cargar_alertas(db)
        finally:
            db.close()

    def _cargar_conductores(self, db) -> None:
        query = text("""
            SELECT id_conductor, nombre, condicion_medica, horario_riesgo
            FROM conductores
        """)
        for row in db.execute(query):
            condiciones = row.condicion_medica or ""
            condiciones_list = ",".join(
                cond.strip().replace(" ", "_")
                for cond in condiciones.split(",")
                if cond.strip()
            )
            horario = (row.horario_riesgo or "dia").replace(" ", "_")
            self.prolog.assertz(
                f"conductor({row.id_conductor}, '{self._sanitize(row.nombre)}', "
                f"[{condiciones_list}], horario({horario}))"
            )

    def _cargar_viajes(self, db) -> None:
        query = text("""
            SELECT id_viaje, id_conductor, id_vehiculo, fecha_fin
            FROM viajes
        """)
        for row in db.execute(query):
            estado = "activo" if row.fecha_fin is None else "finalizado"
            vehiculo = row.id_vehiculo if row.id_vehiculo is not None else "sin_vehiculo"
            self.prolog.assertz(
                f"viaje({row.id_viaje}, conductor({row.id_conductor}), "
                f"vehiculo('{vehiculo}'), estado({estado}))"
            )

    def _cargar_lecturas(self, db) -> None:
        query = text("""
            SELECT id_viaje, percios, conteo_bostezos, conteo_cabeceos, frecuencia_cardiaca
            FROM lecturas_sensores
        """)
        for row in db.execute(query):
            if row.percios is not None:
                self.prolog.assertz(f"lectura_somnolencia({row.id_viaje}, perclos, {row.percios})")
            self.prolog.assertz(f"lectura_somnolencia({row.id_viaje}, bostezos, {row.conteo_bostezos})")
            self.prolog.assertz(f"lectura_somnolencia({row.id_viaje}, cabeceos, {row.conteo_cabeceos})")
            if row.frecuencia_cardiaca is not None:
                self.prolog.assertz(f"lectura_somnolencia({row.id_viaje}, fc, {row.frecuencia_cardiaca})")

    def _cargar_alertas(self, db) -> None:
        query = text("""
            SELECT id_viaje, tipo_alerta, COALESCE(nivel_somnolencia, 'media') AS nivel
            FROM alertas
        """)
        for row in db.execute(query):
            self.prolog.assertz(
                f"alerta_emitida({row.id_viaje}, tipo('{self._sanitize(row.tipo_alerta)}'), nivel({row.nivel}))"
            )

    # ---------------------------------------------------------- consultas
    def evaluar_viaje(self, viaje_id: int) -> Dict:
        self.cargar_desde_db()
        critico = bool(list(self.prolog.query(f"somnolencia_critica({viaje_id})")))
        acciones = [r["Actuador"] for r in self.prolog.query(f"recomendar_accion({viaje_id}, Actuador)")]
        detener = bool(list(self.prolog.query(f"debe_detenerse({viaje_id})")))
        return {
            "viaje": viaje_id,
            "somnolencia_critica": critico,
            "acciones": acciones,
            "debe_detener": detener,
        }

    def viajes_en_riesgo(self) -> List[int]:
        self.cargar_desde_db()
        return [res["Viaje"] for res in self.prolog.query("somnolencia_critica(Viaje)")]


# Instancia única (puede ser reutilizada por FastAPI)
prolog_engine = PrologSomnolenciaEngine()


if __name__ == "__main__":
    # Ejemplo rápido de uso independiente
    engine = PrologSomnolenciaEngine()
    print(engine.evaluar_viaje(1))

