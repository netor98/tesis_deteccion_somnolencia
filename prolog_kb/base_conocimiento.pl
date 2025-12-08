%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Base de conocimiento para el sistema de detección de somnolencia
%% - Contiene hechos estáticos (umbrales, sensores, acciones, etc.)
%% - Los hechos dinámicos (conductores, viajes, lecturas) se cargan desde
%%   PostgreSQL vía PySWIP (ver prolog_engine.py)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% ------------------------------------------------------------------------
%%%%  Hechos estáticos (conocimiento fijo)
%%%% ------------------------------------------------------------------------

% Umbrales PERCLOS
umbral_perclos(leve,     0.20).
umbral_perclos(moderado, 0.35).
umbral_perclos(severo,   0.50).

% Umbrales frecuencia cardiaca
umbral_fc(baja, 60).
umbral_fc(alta, 100).

% Umbrales de eventos de comportamiento (por minuto)
umbral_bostezos(3).
umbral_cabeceos(2).

% Factores de riesgo por horario
factor_riesgo_horario(noche).
factor_riesgo_horario(madrugada).

% Condiciones médicas de riesgo
factor_riesgo_condicion(apnea_del_sueno).
factor_riesgo_condicion(insomnio_cronico).
factor_riesgo_condicion(migrana_cronica).

% Actuadores disponibles para cada nivel de riesgo
accion_alerta(leve,      vibrador).
accion_alerta(moderada,  buzzer).
accion_alerta(severa,    vibrador).
accion_alerta(severa,    buzzer).
accion_alerta(severa,    notificacion_supervisor).

% Sensores y métricas que ofrecen
sensor_metrica(camara, perclos).
sensor_metrica(camara, bostezos).
sensor_metrica(pulso,  fc).
sensor_metrica(pir,    cabeceos).


%%%% ------------------------------------------------------------------------
%%%%  Hechos dinámicos (se insertan desde prolog_engine.py)
%%%% ------------------------------------------------------------------------
% conductor(Id, Nombre, ListaCondiciones, horario(Periodo)).
% viaje(IdViaje, conductor(IdConductor), vehiculo('AAA-123'), estado(activo/finalizado)).
% lectura_somnolencia(Viaje, Metrica, Valor).
% alerta_emitida(Viaje, tipo('HEAD_TILT'), nivel(alto)).
%
% Estos hechos NO se escriben aquí, sino que se cargan con assertz/1
% para mantener sincronía con el SGBD.


%%%% ------------------------------------------------------------------------
%%%%  Reglas (al menos 8, aquí hay 9)
%%%% ------------------------------------------------------------------------

riesgo_perclos(Viaje) :-
    lectura_somnolencia(Viaje, perclos, Valor),
    umbral_perclos(severo, Umbral),
    Valor >= Umbral.

riesgo_bostezos(Viaje) :-
    lectura_somnolencia(Viaje, bostezos, Valor),
    umbral_bostezos(Limite),
    Valor >= Limite.

riesgo_cabeceos(Viaje) :-
    lectura_somnolencia(Viaje, cabeceos, Valor),
    umbral_cabeceos(Limite),
    Valor >= Limite.

riesgo_fc(Viaje) :-
    lectura_somnolencia(Viaje, fc, Valor),
    (Valor =< umbral_fc(baja, B) ; Valor >= umbral_fc(alta, A)).

riesgo_combinado(Viaje) :-
    riesgo_perclos(Viaje) ;
    riesgo_bostezos(Viaje) ;
    riesgo_cabeceos(Viaje).

somnolencia_alta(Viaje) :-
    riesgo_perclos(Viaje),
    riesgo_bostezos(Viaje).

somnolencia_critica(Viaje) :-
    somnolencia_alta(Viaje),
    riesgo_cabeceos(Viaje).

recomendar_accion(Viaje, Actuador) :-
    somnolencia_critica(Viaje),
    accion_alerta(severa, Actuador).

recomendar_accion(Viaje, Actuador) :-
    somnolencia_alta(Viaje),
    accion_alerta(moderada, Actuador).

debe_detenerse(Viaje) :-
    somnolencia_critica(Viaje),
    \+ alerta_emitida(Viaje, tipo('STOP')).

%%%% ------------------------------------------------------------------------
%%%%  Fin del archivo
%%%% ------------------------------------------------------------------------

