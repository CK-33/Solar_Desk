"""Ensemble versionado; treino, early stopping, stacking, calibração e teste separados."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solar_core import features, local_label, STEP


def metrics(y, pred, poa):
    mask = (poa > 20) & y.notna() & pred.notna()
    if mask.sum() < 20:
        raise ValueError('Poucas amostras diurnas para avaliar o modelo (mínimo: 20).')
    real, est = y[mask], pred[mask]
    error = real - est
    denom = float(((real - real.mean()) ** 2).sum())
    return {'mae': float(error.abs().mean()), 'rmse': float(np.sqrt((error ** 2).mean())),
            'r2': float(1 - (error ** 2).sum() / denom) if denom > 0 else None, 'n': int(mask.sum())}


def load_info(home, version):
    return json.loads((Path(home) / 'models' / version / 'info.json').read_text(encoding='utf-8'))


class Model:
    def __init__(self, home, version):
        import xgboost as xgb
        from catboost import CatBoostRegressor
        path = Path(home) / 'models' / version
        self.info = load_info(home, version)
        self.xgb = xgb.Booster()
        self.xgb.load_model(path / 'xgb.ubj')
        self.xgb.set_param({'nthread': 2})
        self.cb = CatBoostRegressor()
        self.cb.load_model(str(path / 'catboost.cbm'))

    def predict(self, x):
        import xgboost as xgb
        if x.empty:
            return pd.Series(dtype=float, index=x.index, name='previsto')
        xp = np.maximum(self.xgb.predict(xgb.DMatrix(x), iteration_range=(0, self.info['best_iteration'] + 1)), 0)
        cp = np.maximum(self.cb.predict(x, thread_count=2), 0)
        pred = np.maximum(self.info['intercept'] + np.column_stack([xp, cp]) @ np.array(self.info['coef']), 0)
        pred[x.POA.to_numpy() < 5] = 0
        return pd.Series(pred, index=x.index, name='previsto')


def train_candidate(store, log=lambda text: None, rounds=1200):
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge

    df = store.read()
    x = features(df)
    if len(x) < 96 * 30:
        raise ValueError('São necessários pelo menos 30 dias de amostras utilizáveis para separar as etapas temporais.')
    y = df.pdc.reindex(x.index)
    old_version = store.setting('active_model')
    test_start = int(len(x) * .9)
    if old_version:
        old = load_info(store.home, old_version)
        after = int(x.index.searchsorted(pd.Timestamp(old['evaluated_until']), side='right'))
        test_start = max(test_start, after)
    if len(x) - test_start < 96 * 7:
        raise ValueError('Importe pelo menos 7 dias novos para avaliar a atualização em dados ainda não usados na escolha do modelo.')
    # Frações do histórico anterior ao teste: 2/3 base, 1/9 early, 1/9 meta, 1/9 calibração.
    a, b, c = [int(test_start * fraction) for fraction in [2/3, 7/9, 8/9]]
    if min(a, b-a, c-b, test_start-c) < 96 * 2:
        raise ValueError('Histórico insuficiente para os blocos independentes de treinamento.')
    blocks = {'base': (0, a), 'early_stopping': (a, b), 'stacking': (b, c),
              'calibration': (c, test_start), 'test': (test_start, len(x))}
    log(f'Treinamento: {a:,} amostras; teste reservado: {len(x)-test_start:,}.')
    log('Treinando XGBoost em CPU (até 4 threads)...')
    dm = xgb.DMatrix(x.iloc[:a], label=y.iloc[:a])
    ev = xgb.DMatrix(x.iloc[a:b], label=y.iloc[a:b])
    xb = xgb.train({'objective': 'reg:absoluteerror', 'eval_metric': 'mae', 'tree_method': 'hist',
                    'max_depth': 6, 'learning_rate': .04, 'subsample': .8, 'colsample_bytree': .85,
                    'min_child_weight': 20, 'reg_lambda': 1., 'seed': 42, 'nthread': 4}, dm,
                   num_boost_round=rounds, evals=[(ev, 'early')], early_stopping_rounds=80, verbose_eval=False)
    log('Treinando CatBoost...')
    cb = CatBoostRegressor(iterations=rounds, depth=6, learning_rate=.04, l2_leaf_reg=3,
                           loss_function='MAE', random_seed=42, thread_count=4,
                           early_stopping_rounds=80, allow_writing_files=False, verbose=False)
    cb.fit(x.iloc[:a], y.iloc[:a], eval_set=(x.iloc[a:b], y.iloc[a:b]), use_best_model=True)
    def bases(z):
        return np.column_stack([np.maximum(xb.predict(xgb.DMatrix(z), iteration_range=(0, xb.best_iteration + 1)), 0),
                                np.maximum(cb.predict(z, thread_count=4), 0)])
    log('Ajustando Ridge em um bloco independente...')
    ridge = Ridge(alpha=1.)
    ridge.fit(bases(x.iloc[b:c]), y.iloc[b:c])
    def predict(z):
        p = np.maximum(ridge.predict(bases(z)), 0)
        p[z.POA.to_numpy() < 5] = 0
        return pd.Series(p, index=z.index)
    cal = x.iloc[c:test_start]
    residual = y.reindex(cal.index) - predict(cal)
    day_res = residual[cal.POA > 20]
    if len(day_res) < 20:
        raise ValueError('Calibração sem amostras diurnas suficientes.')
    sigma = float(day_res.std())
    if not np.isfinite(sigma):
        raise ValueError('Não foi possível calibrar os resíduos.')
    test = x.iloc[test_start:]
    pred = predict(test)
    score = metrics(y.reindex(test.index), pred, test.POA)
    persistence = metrics(y.reindex(test.index), test.Pdc1_lag_1, test.POA)
    incumbent = None
    if old_version:
        log('Comparando com a versão em uso nos mesmos novos horários...')
        incumbent = metrics(y.reindex(test.index), Model(store.home, old_version).predict(test), test.POA)
    eligible = score['mae'] <= persistence['mae'] and (incumbent is None or score['mae'] <= incumbent['mae'])
    version = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    path = store.home / 'models' / version
    path.mkdir(parents=True)
    info = {'version': version, 'created': datetime.now().isoformat(timespec='seconds'),
            'trained_through': x.index[test_start-1].isoformat(), 'evaluated_until': x.index[-1].isoformat(),
            'features': list(x.columns), 'cadence_minutes': 15, 'best_iteration': xb.best_iteration,
            'coef': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_), 'sigma': sigma,
            'metrics': score, 'persistence': persistence, 'incumbent': incumbent,
            'previous_version': old_version, 'eligible': bool(eligible), 'rows': len(x),
            'periods': {name: {'start': x.index[i].isoformat(), 'end': x.index[j-1].isoformat(), 'rows': j-i} for name,(i,j) in blocks.items()},
            'data_origin': 'Base fornecida pelo usuário; procedência e códigos Validacao não confirmados.',
            'versions': {'xgboost': xgb.__version__}}
    xb.save_model(path / 'xgb.ubj')
    cb.save_model(str(path / 'catboost.cbm'))
    (path / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report = pd.DataFrame({'real_W': y.reindex(test.index), 'previsto_W': pred, 'persistencia_W': test.Pdc1_lag_1, 'POA_Wm2': test.POA})
    report.to_csv(path / 'teste.csv', encoding='utf-8-sig')
    store.set_setting('candidate_model', version)
    log(f"Candidata salva. MAE diurno: {score['mae']:.2f} W; persistência: {persistence['mae']:.2f} W.")
    log('Disponível para ativação.' if eligible else 'A candidata não superou a referência; a versão em uso foi preservada.')
    return info


def activate_candidate(store):
    version = store.setting('candidate_model')
    if not version:
        raise ValueError('Não há candidata para ativar.')
    info = load_info(store.home, version)
    if not info['eligible']:
        raise ValueError('A candidata não passou pelo critério de MAE frente às referências.')
    current = store.setting('active_model')
    if current == version:
        return info
    if current != info['previous_version']:
        raise ValueError('A versão em uso mudou após a avaliação. Treine uma nova candidata.')
    # Valida os arquivos antes de alterar o ponteiro no banco.
    Model(store.home, version)
    with store.connect() as con:
        for k, v in [('previous_model', current), ('active_model', version)]:
            con.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (k, json.dumps(v)))
    return info


def rollback(store):
    previous = store.setting('previous_model')
    if not previous:
        raise ValueError('Ainda não existe uma versão anterior para restaurar.')
    Model(store.home, previous)
    with store.connect() as con:
        current = store.setting('active_model')
        for k, v in [('active_model', previous), ('previous_model', current)]:
            con.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (k, json.dumps(v)))


def analysis(store, log=lambda text: None):
    from solar_core import data_summary, physical, detect_alerts
    df = store.read()
    summary = data_summary(df)
    physical_values, daily = physical(df, store.setting('nominal_w'))
    version = store.setting('active_model')
    pred = pd.Series(dtype=float, name='previsto')
    info = None
    if version and not df.empty:
        log('Atualizando previsões com o modelo salvo...')
        model = Model(store.home, version)
        info = model.info
        x = features(df)
        # Exibição operacional apenas depois do período usado no ajuste/calibração.
        x = x[x.index > pd.Timestamp(info['trained_through'])]
        pred = model.predict(x)
    alerts = detect_alerts(df, pred if info else None, info['sigma'] if info else None)
    return {'df': df, 'summary': summary, 'physical': physical_values, 'daily': daily,
            'prediction': pred, 'model': info, 'alerts': alerts, 'imports': store.import_log()}
