"""
Pequeño script de ejemplo para invocar el motor Prolog sin pasar por FastAPI.
Útil para pruebas manuales:

    python prolog_kb/demo_usage.py --viaje 101
"""

import argparse

from prolog_kb.prolog_engine import PrologSomnolenciaEngine


def main(viaje_id: int):
    engine = PrologSomnolenciaEngine()
    resultado = engine.evaluar_viaje(viaje_id)
    print(f"Resultado Prolog para viaje {viaje_id}:")
    print(resultado)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viaje", type=int, required=True)
    args = parser.parse_args()
    main(args.viaje)

