"""Importação transacional e indicadores locais. Nenhum acesso à rede."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager, closing
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = timezone(timedelta(hours=-3))
STEP = pd.Timedelta(minutes=15)
COLS = ['poa', 'pdc', 'vdc', 'idc', 'validation', 'temp_air']


def utc_stamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize(LOCAL_TZ).tz_convert('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def local_label(value, fmt='%d/%m/%Y %H:%M'):
    return utc_stamp(value).tz_convert(LOCAL_TZ).strftime(fmt)


class Store:
    def __init__(self, home):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.db = self.home / 'historico.sqlite3'
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS readings (
                    ts TEXT PRIMARY KEY, poa REAL NOT NULL, pdc REAL NOT NULL,
                    vdc REAL, idc REAL, validation TEXT, temp_air REAL, source TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS imports (
                    id INTEGER PRIMARY KEY, created TEXT, filename TEXT, sha256 TEXT,
                    added INTEGER, duplicates INTEGER);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            ''')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=30)
        try:
            con.execute('PRAGMA journal_mode=WAL')
            with con:
                yield con
        finally:
            con.close()

    def setting(self, key, default=None):
        with self.connect() as con:
            row = con.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.connect() as con:
            con.execute('INSERT OR REPLACE INTO settings VALUES (?, ?)', (key, json.dumps(value)))

    def read(self):
        with self.connect() as con:
            df = pd.read_sql_query('SELECT * FROM readings ORDER BY ts', con)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop('ts'), utc=True), name='Datetime')
        for c in COLS:
            if c != 'validation':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def import_log(self):
        with self.connect() as con:
            return pd.read_sql_query('SELECT created,filename,added,duplicates FROM imports ORDER BY id DESC', con)

    def import_csv(self, path):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.connect() as con:
            if con.execute('SELECT 1 FROM imports WHERE sha256=?', (digest,)).fetchone():
                return {'file': path.name, 'added': 0, 'duplicates': 0, 'unchanged': True}
        df = parse_csv(path)
        within = int(df.index.duplicated().sum())
        for _, group in df[df.index.duplicated(keep=False)].groupby(level=0):
            if len(group.drop_duplicates()) > 1:
                raise ValueError('Há horários repetidos com valores diferentes no arquivo. Nenhum dado foi importado.')
        df = df[~df.index.duplicated()].sort_index()
        duplicates = within
        added = 0
        # Arquivo inteiro confirmado ou revertido; registros existentes nunca são sobrescritos.
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            for ts, row in df.iterrows():
                key = ts.isoformat()
                values = tuple(None if pd.isna(row[c]) else row[c] for c in COLS)
                old = con.execute('SELECT poa,pdc,vdc,idc,validation,temp_air FROM readings WHERE ts=?', (key,)).fetchone()
                if old is not None:
                    same = all(a == b or (a is not None and b is not None and c != 'validation' and np.isclose(float(a), float(b), rtol=1e-8, atol=1e-8)) for a, b, c in zip(old, values, COLS))
                    if not same:
                        raise ValueError(f'Conflito em {local_label(ts)}: o histórico tem valores diferentes. O arquivo foi revertido; revise a origem.')
                    duplicates += 1
                else:
                    con.execute('INSERT INTO readings VALUES (?,?,?,?,?,?,?,?)', (key, *values, path.name))
                    added += 1
            con.execute('INSERT INTO imports(created,filename,sha256,added,duplicates) VALUES (?,?,?,?,?)',
                        (datetime.now().isoformat(timespec='seconds'), path.name, digest, added, duplicates))
        return {'file': path.name, 'added': added, 'duplicates': duplicates, 'unchanged': False}

    def backup(self, destination):
        if Path(destination).resolve() == self.db.resolve():
            raise ValueError('Escolha outro caminho para o backup.')
        with self.connect() as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)


def parse_csv(path):
    try:
        raw = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig', dtype=str)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, sep=None, engine='python', encoding='cp1252', dtype=str)
    raw.columns = raw.columns.str.strip()
    required = {'Datetime', 'POA', 'Pdc1'}
    if not required.issubset(raw.columns):
        raise ValueError('Formato não reconhecido. Colunas obrigatórias: Datetime, POA, Pdc1. O formato Time/Rg Média precisa de um adaptador confirmado.')
    if raw.empty:
        raise ValueError('O CSV está vazio.')
    times = []
    for i, value in enumerate(raw['Datetime']):
        try:
            ts = utc_stamp(value)
            if pd.isna(ts):
                raise ValueError()
            times.append(ts)
        except Exception as exc:
            raise ValueError(f'Data inválida na linha {i + 2}. Use ISO 8601; sem fuso, assume-se UTC−03.') from exc
    index = pd.DatetimeIndex(times)
    if ((index.minute % 15 != 0) | (index.second != 0) | (index.microsecond != 0) | (index.nanosecond != 0)).any():
        raise ValueError('Esta versão atende dados alinhados a intervalos de 15 minutos. Não houve reamostragem automática.')
    df = pd.DataFrame(index=index)
    mapping = {'POA': 'poa', 'Pdc1': 'pdc', 'Vdc1': 'vdc', 'Idc1': 'idc', 'Tar med': 'temp_air'}
    for original, c in mapping.items():
        if original in raw:
            s = raw[original].str.strip().str.replace(',', '.', regex=False)
            values = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
            missing = s.isna().to_numpy() | s.fillna('').eq('').to_numpy()
            if np.isinf(values).any() or (np.isnan(values) & ~missing).any() or (c in ['poa', 'pdc'] and np.isnan(values).any()):
                raise ValueError(f'Valores ausentes ou não numéricos em {original}. Revise o CSV antes de importar.')
            if c != 'temp_air' and (values < 0).any():
                raise ValueError(f'Valores negativos em {original}. O arquivo foi recusado para revisão.')
            df[c] = values
        else:
            df[c] = np.nan
    df['validation'] = raw['Validacao'].fillna('').str.strip().to_numpy() if 'Validacao' in raw else None
    return df[COLS]


def data_summary(df):
    if df.empty:
        return {'rows': 0}
    gaps = df.index.to_series().diff()
    missing = int(((gaps[gaps > STEP] / STEP) - 1).sum())
    return {'rows': len(df), 'start': local_label(df.index.min()), 'end': local_label(df.index.max()),
            'missing': missing, 'gaps': int((gaps > STEP).sum()), 'max_power': float(df.pdc.max()),
            'energy_kwh': float(df.pdc.sum() * 0.25 / 1000),
            'validation': df.validation.value_counts(dropna=False).to_dict()}


def physical(df, nominal=None):
    """PR ponderado por irradiância. Energia é estimativa: amostras × 0,25 h."""
    result = pd.DataFrame(index=df.index)
    result['pr'] = np.nan
    day = df.poa > 50
    if nominal and nominal > 0:
        result.loc[day, 'pr'] = df.loc[day, 'pdc'] / (nominal * df.loc[day, 'poa'] / 1000)
    # FF não calculado sem parâmetros do equipamento e temperatura confirmados.
    daily = df[['pdc', 'poa']].copy()
    daily.index = daily.index.tz_convert(LOCAL_TZ)
    agg = daily.resample('D').agg(pdc_sum=('pdc', 'sum'), poa_sum=('poa', 'sum'), samples=('pdc', 'count'))
    agg['energy_kwh'] = agg.pdc_sum * .25 / 1000
    agg['coverage'] = agg.samples / 96
    agg['pr'] = np.nan
    if nominal and nominal > 0:
        solar = daily[daily.poa > 50].resample('D')[['pdc', 'poa']].sum()
        agg['pr'] = solar.pdc / (nominal * solar.poa.replace(0, np.nan) / 1000)
    agg['pr_7d'] = agg.pr.rolling('7D', min_periods=3).mean()
    agg['pr_delta_7d'] = agg.pr_7d.diff(7)
    return result, agg


def detect_alerts(df, forecast=None, sigma=None):
    events = []
    if df.empty:
        return events
    for ts, gap in df.index.to_series().diff().items():
        if gap > STEP:
            events.append({'date': local_label(ts), 'type': 'Dados', 'detail': f'Lacuna de {gap / pd.Timedelta(hours=1):.2f} h entre registros.', 'ts': ts})

    def groups(mask, kind, text, min_samples):
        prev = None
        run = []
        def flush():
            if len(run) >= min_samples:
                duration = (run[-1] - run[0]) / pd.Timedelta(hours=1)
                events.append({'date': local_label(run[0]), 'type': kind,
                               'detail': f'{text} {len(run)} amostras; intervalo observado de {duration:.2f} h.', 'ts': run[0]})
        for ts, flagged in mask.items():
            if not flagged or (prev is not None and ts - prev != STEP):
                flush()
                run = []
            if flagged:
                run.append(ts)
            prev = ts
        flush()

    groups((df.poa > 200) & (df.pdc < 10), 'Operação', 'Possível falha: irradiância > 200 W/m² e potência < 10 W.', 13)
    if forecast is not None and sigma is not None and sigma > 0:
        # Somente previsões após os dados de ajuste/calibração; jamais resíduos de treino.
        aligned = forecast.reindex(df.index)
        groups((df.pdc - aligned < -2 * sigma) & (df.poa > 20), 'Desvio', 'Geração abaixo da referência por mais de 2σ; causa não determinada.', 4)
    return sorted(events, key=lambda r: r['ts'], reverse=True)


def features(df):
    x = pd.DataFrame(index=df.index)
    local = df.index.tz_convert(LOCAL_TZ)
    hour = local.hour + local.minute / 60
    x['POA'] = df.poa
    x['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    x['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    x['doy_sin'] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    x['doy_cos'] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    # Busca pelo horário exato: uma lacuna não encurta artificialmente o lag.
    for lag in [1, 2, 4, 8, 12, 24]:
        x[f'POA_lag_{lag}'] = df.poa.reindex(df.index - lag * STEP).to_numpy()
    for lag in [1, 2]:
        x[f'Pdc1_lag_{lag}'] = df.pdc.reindex(df.index - lag * STEP).to_numpy()
    window = pd.concat([df.poa.reset_index(drop=True)] + [pd.Series(df.poa.reindex(df.index - lag * STEP).to_numpy()) for lag in [1, 2, 3]], axis=1)
    x['POA_roll_mean_4'] = window.mean(axis=1, skipna=False).to_numpy()
    return x.dropna()
