"""SolarDesk — aplicativo local para Windows."""
from __future__ import annotations

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# Carrega bibliotecas científicas antes do hook de imports do PySide.
# Evita inspeção de milhares de módulos científicos na primeira tarefa de fundo.
import pandas.io.sql
import xgboost
import catboost
from sklearn.linear_model import Ridge
from PySide6.QtCore import QDate, QLockFile, QThread, QTimer, Qt, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDateEdit, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from solar_core import Store, LOCAL_TZ, local_label
from solar_ml import analysis, train_candidate, activate_candidate, load_info, rollback

ROOT = Path(__file__).resolve().parent
BG = '#f4f6f3'
INK = '#18392d'
GREEN = '#167153'
MUTED = '#75847b'

STYLE = '''
QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #213e32; }
QMainWindow, #workspace { background: #f4f6f3; }
#sidebar { background: #153d30; border-radius: 0; }
#sidebar QLabel { color: #d5e5dc; background: transparent; }
#brand { font-size: 25px; font-weight: 700; color: white; }
#sideSub { color: #97b7a5; font-size: 11px; }
#sidebar QPushButton { background: transparent; color: #b5d0c0; border: 0; text-align: left; padding: 13px 16px; border-radius: 8px; }
#sidebar QPushButton:hover { background: #214e3b; }
#sidebar QPushButton:checked { background: #d9f3a5; color: #173b2c; font-weight: 700; }
#title { font-size: 29px; font-weight: 650; color: #163d2c; }
#subtitle, #muted { color: #75847b; }
#eyebrow { color: #728176; font-size: 10px; font-weight: 700; }
#card, #panel { background: white; border: 1px solid #e0e7df; border-radius: 12px; }
#value { font-size: 28px; font-weight: 650; color: #143e2c; }
#panelTitle { font-size: 17px; font-weight: 650; }
#notice { background: #ecf1e8; color: #536548; padding: 13px; border-radius: 8px; }
QPushButton { background: white; border: 1px solid #d3ded3; border-radius: 7px; padding: 9px 14px; font-weight: 600; }
QPushButton:hover { background: #edf5e6; }
QPushButton:disabled { background: #eff1ed; color: #98a598; border-color: #e0e6de; }
QPushButton#primary { background: #1a704f; border: 1px solid #1a704f; color: white; }
QPushButton#primary:hover { background: #13583c; }
QPushButton#primary:disabled { background: #aec5b6; border-color: #aec5b6; }
QDateEdit, QDoubleSpinBox { background: white; border: 1px solid #d9e1d6; padding: 7px; border-radius: 6px; }
QTableWidget { background: white; border: 0; gridline-color: #eef2ec; alternate-background-color: #f7faf5; selection-background-color: #dceccc; selection-color: #17372b; }
QHeaderView::section { background: #f3f7ef; border: 0; padding: 10px; color: #667961; font-weight: 600; }
QTableWidget::item { padding: 8px; }
QPlainTextEdit { background: #f5f8f2; border: 1px solid #dfe7da; border-radius: 8px; padding: 8px; }
QProgressBar { background: #e6eddf; border: 0; border-radius: 3px; height: 5px; }
QProgressBar::chunk { background: #78a553; border-radius: 3px; }
QToolBar { background: white; border: 0; spacing: 5px; }
QScrollArea { border: 0; background: transparent; }
'''


def apply_theme(app):
    # Registro explícito também permite testes gráficos fora da tela no Windows.
    font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ['segoeui.ttf', 'segoeuib.ttf', 'seguisb.ttf']:
        path = font_dir / name
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    app.setFont(QFont('Segoe UI', 10))


def label(text='', name=None, wrap=False):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def panel():
    w = QFrame()
    w.setObjectName('panel')
    layout = QVBoxLayout(w)
    layout.setContentsMargins(22, 18, 22, 18)
    layout.setSpacing(12)
    return w, layout


def table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().hide()
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.setWordWrap(True)
    return t


def fill_table(t, rows):
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            item = QTableWidgetItem(str(v))
            item.setToolTip(str(v))
            t.setItem(i, j, item)
    t.resizeRowsToContents()


class Worker(QThread):
    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.result.emit(self.fn(self.progress.emit))
        except Exception as exc:
            logging.exception('Falha em operação')
            self.error.emit(str(exc))


class Window(QMainWindow):
    def __init__(self, store, autoload=True):
        super().__init__()
        self.store = store
        self.worker = None
        self.state = None
        self.callback = None
        self.date_initialized = False
        self.actions = []
        self.setWindowTitle('SolarDesk | Monitoramento fotovoltaico')
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1380, screen.width() - 60), min(920, screen.height() - 60))
        self.setMinimumSize(980, 640)
        self.setWindowIcon(self.make_icon())
        base = QWidget()
        base.setObjectName('workspace')
        self.setCentralWidget(base)
        outer = QHBoxLayout(base)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(208)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 30, 20, 25)
        side.addWidget(label('SolarDesk', 'brand'))
        side.addWidget(label('INTELIGÊNCIA FOTOVOLTAICA', 'sideSub'))
        side.addSpacing(35)
        self.navigation = QButtonGroup(self)
        names = ['Visão geral', 'Dados da usina', 'Nowcasting', 'Alertas', 'Configurações']
        for i, name in enumerate(names):
            button = QPushButton(f'{i+1:02}   {name}')
            button.setCheckable(True)
            self.navigation.addButton(button, i)
            side.addWidget(button)
        self.navigation.button(0).setChecked(True)
        self.navigation.idClicked.connect(self.navigate)
        side.addStretch()
        side.addWidget(label('UMA BASE. DUAS ANÁLISES.', 'sideSub'))
        side.addWidget(label('Geração esperada\nIndicadores de desempenho', wrap=True))
        side.addSpacing(15)
        side.addWidget(label('Versão inicial • armazenamento local\nUTC−03  /  intervalos de 15 min', 'sideSub', True))
        outer.addWidget(sidebar)
        main = QVBoxLayout()
        main.setContentsMargins(30, 25, 30, 18)
        main.setSpacing(14)
        outer.addLayout(main, 1)
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(label('MONITORAMENTO  /  BASE DISPONÍVEL', 'eyebrow'))
        self.title = label('Visão geral', 'title')
        title_box.addWidget(self.title)
        self.subtitle = label('Seu histórico de geração, em um só lugar.', 'subtitle')
        title_box.addWidget(self.subtitle)
        header.addLayout(title_box, 1)
        self.import_button = self.button('+  Importar CSV', self.import_files, primary=True)
        header.addWidget(self.import_button)
        main.addLayout(header)
        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        self.build_overview()
        self.build_data()
        self.build_models()
        self.build_alerts()
        self.build_settings()
        self.status = label('Carregando histórico local...', 'muted')
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        self.progress.hide()
        main.addWidget(self.status)
        main.addWidget(self.progress)
        if autoload:
            QTimer.singleShot(100, self.refresh)

    @staticmethod
    def make_icon():
        pix = QPixmap(64, 64)
        pix.fill(QColor('#153d30'))
        p = QPainter(pix)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor('#d9f3a5'))
        p.drawEllipse(16, 16, 32, 32)
        p.end()
        return QIcon(pix)

    def button(self, text, callback, primary=False):
        b = QPushButton(text)
        if primary:
            b.setObjectName('primary')
        b.clicked.connect(callback)
        self.actions.append(b)
        return b

    def page(self):
        p = QWidget()
        p.setObjectName('workspace')
        box = QVBoxLayout(p)
        box.setContentsMargins(0, 6, 0, 0)
        box.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(p)
        self.stack.addWidget(scroll)
        return box

    def build_overview(self):
        box = self.page()
        cards = QHBoxLayout()
        self.card_values = []
        self.card_notes = []
        for title, note in [('ENERGIA NO PERÍODO', 'Estimativa pelas amostras'), ('PICO NO PERÍODO', 'Potência DC medida'),
                            ('ERRO DO MODELO', 'MAE diurno • teste reservado'), ('HISTÓRICO LOCAL', 'Registros únicos')]:
            w, l = panel()
            w.setObjectName('card')
            l.addWidget(label(title, 'eyebrow'))
            value = label('—', 'value')
            l.addWidget(value)
            sub = label(note, 'muted', True)
            l.addWidget(sub)
            self.card_values.append(value)
            self.card_notes.append(sub)
            cards.addWidget(w, 1)
        box.addLayout(cards)
        chart_panel, lay = panel()
        row = QHBoxLayout()
        row.addWidget(label('Geração real e esperada', 'panelTitle'), 1)
        self.start_date, self.end_date = QDateEdit(), QDateEdit()
        for edit in [self.start_date, self.end_date]:
            edit.setCalendarPopup(True)
            edit.setDisplayFormat('dd/MM/yyyy')
            edit.setFixedWidth(118)
        row.addWidget(self.start_date)
        row.addWidget(label('até', 'muted'))
        row.addWidget(self.end_date)
        self.apply_dates = self.button('Aplicar', self.plot)
        row.addWidget(self.apply_dates)
        lay.addLayout(row)
        self.chart_note = label('Importe sua base para começar.', 'muted', True)
        lay.addWidget(self.chart_note)
        self.figure = Figure(figsize=(10, 4), facecolor='white', layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(260)
        lay.addWidget(self.canvas, 1)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setMaximumHeight(30)
        lay.addWidget(toolbar)
        box.addWidget(chart_panel, 1)
        self.overview_note = label('PR aguarda potência nominal confirmada. FF aguarda parâmetros e temperatura.', 'notice', True)
        box.addWidget(self.overview_note)

    def build_data(self):
        box = self.page()
        p, l = panel()
        l.addWidget(label('Uma importação organizada, sem duplicar seu histórico', 'panelTitle'))
        l.addWidget(label('Formato atual: Datetime, POA e Pdc1. Vdc1, Idc1, Validacao e Tar med são opcionais. '
                          'Dados sem fuso são interpretados em UTC−03. Conflitos não sobrescrevem registros existentes.', 'muted', True))
        self.data_note = label('', 'notice', True)
        l.addWidget(self.data_note)
        buttons = QHBoxLayout()
        buttons.addWidget(self.button('Exportar histórico CSV', self.export_history))
        buttons.addWidget(self.button('Backup do histórico', self.backup))
        buttons.addStretch()
        l.addLayout(buttons)
        box.addWidget(p)
        p, l = panel()
        l.addWidget(label('Últimos registros • até 500 linhas', 'panelTitle'))
        self.data_table = table(['Data / hora (UTC−03)', 'Potência DC (W)', 'POA (W/m²)', 'Código original'])
        l.addWidget(self.data_table)
        box.addWidget(p, 2)
        p, l = panel()
        l.addWidget(label('Histórico de importações', 'panelTitle'))
        self.import_table = table(['Importado em', 'Arquivo', 'Novos', 'Repetidos'])
        l.addWidget(self.import_table)
        box.addWidget(p, 1)

    def build_models(self):
        box = self.page()
        p, l = panel()
        l.addWidget(label('Nowcasting com XGBoost + CatBoost', 'panelTitle'))
        l.addWidget(label('O modelo estima a potência do instante observado usando irradiância atual e histórico recente. '
                          'Cada importação atualiza as previsões; o aprendizado é atualizado pelo botão abaixo.', 'muted', True))
        self.model_status = label('Nenhum modelo ativo.', 'notice', True)
        l.addWidget(self.model_status)
        buttons = QHBoxLayout()
        self.train_button = self.button('Treinar nova versão', self.train, True)
        self.activate_button = self.button('Ativar candidata aprovada', self.activate)
        self.rollback_button = self.button('Restaurar versão anterior', self.restore)
        for b in [self.train_button, self.activate_button, self.rollback_button]:
            buttons.addWidget(b)
        buttons.addStretch()
        l.addLayout(buttons)
        l.addWidget(label('Treino, parada antecipada, combinação, calibração e teste usam blocos cronológicos separados. '
                          'A candidata precisa igualar ou melhorar o MAE da persistência e da versão em uso no mesmo teste.', 'muted', True))
        box.addWidget(p)
        p, l = panel()
        l.addWidget(label('Última avaliação', 'panelTitle'))
        self.model_table = table(['Referência', 'MAE (W)', 'RMSE (W)', 'R²', 'Amostras diurnas'])
        self.model_table.setMaximumHeight(175)
        l.addWidget(self.model_table)
        self.model_detail = label('', 'muted', True)
        l.addWidget(self.model_detail)
        box.addWidget(p)
        p, l = panel()
        l.addWidget(label('Atividade', 'panelTitle'))
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        l.addWidget(self.log_box)
        box.addWidget(p, 1)

    def build_alerts(self):
        box = self.page()
        p, l = panel()
        l.addWidget(label('Evidências para investigar', 'panelTitle'))
        l.addWidget(label('Os alertas indicam lacunas, possível falha operacional ou geração persistentemente abaixo da referência. '
                          'Eles não confirmam sujidade ou necessidade de limpeza.', 'muted', True))
        self.alert_note = label('', 'notice', True)
        l.addWidget(self.alert_note)
        self.alert_table = table(['Início / detecção', 'Categoria', 'Evidência observada'])
        self.alert_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.alert_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        l.addWidget(self.alert_table, 1)
        l.addWidget(self.button('Exportar alertas CSV', self.export_alerts))
        box.addWidget(p, 1)

    def build_settings(self):
        box = self.page()
        p, l = panel()
        l.addWidget(label('Configuração da base analisada', 'panelTitle'))
        l.addWidget(label('A potência nominal deve corresponder exatamente ao equipamento representado por Pdc1. '
                          'Não use a potência total da usina se a coluna representar apenas um módulo ou uma string.', 'muted', True))
        row = QHBoxLayout()
        row.addWidget(label('Potência nominal DC (W)'))
        self.nominal = QDoubleSpinBox()
        self.nominal.setRange(0, 100000000)
        self.nominal.setDecimals(2)
        self.nominal.setSpecialValueText('Não confirmada')
        self.nominal.setValue(self.store.setting('nominal_w') or 0)
        row.addWidget(self.nominal)
        row.addWidget(self.button('Salvar e recalcular', self.save_settings, True))
        row.addStretch()
        l.addLayout(row)
        l.addWidget(label('PR: disponível após confirmar a potência nominal; sem corte artificial em 1,1. '
                          'FF: indisponível nesta versão porque os parâmetros elétricos e a temperatura necessários não foram confirmados.', 'notice', True))
        box.addWidget(p)
        p, l = panel()
        l.addWidget(label('O que esta primeira versão conhece', 'panelTitle'))
        l.addWidget(label('• Base inicial: 2020, em intervalos de 15 minutos.\n'
                          '• POA e potência em W/m² e W, conforme o código recebido.\n'
                          '• Códigos Validacao preservados, sem filtragem automática.\n'
                          '• Procedência medida/simulada ainda não confirmada.\n'
                          '• Energia estimada pela soma das amostras × 0,25 h; lacunas não são preenchidas.\n'
                          '• Alerta operacional: POA > 200 W/m² e Pdc1 < 10 W durante pelo menos 3 h observadas.\n'
                          '• Alerta de desvio: resíduo < −2σ em pelo menos 4 amostras consecutivas diurnas.', 'muted', True))
        box.addWidget(p)
        p, l = panel()
        l.addWidget(label('Arquivos locais', 'panelTitle'))
        l.addWidget(label(str(self.store.home.resolve()), 'muted', True))
        l.addWidget(self.button('Abrir pasta de dados e modelos', lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.home.resolve())))))
        l.addWidget(label('O backup do histórico inclui leituras, importações e configurações. Para preservar também os modelos, '
                          'copie a pasta data inteira com o aplicativo fechado.', 'muted', True))
        box.addWidget(p)
        box.addStretch()

    def navigate(self, index):
        self.stack.setCurrentIndex(index)
        self.navigation.button(index).setChecked(True)
        self.title.setText(['Visão geral', 'Dados da usina', 'Nowcasting', 'Alertas', 'Configurações'][index])
        self.subtitle.setText(['Seu histórico de geração, em um só lugar.', 'Importe, confira e preserve suas medições.',
                               'Uma referência de geração que você pode avaliar.', 'Observe os desvios. Investigue as causas.',
                               'Parâmetros explícitos para resultados confiáveis.'][index])

    def log(self, text):
        self.status.setText(text)
        self.status.setToolTip(text)
        self.log_box.appendPlainText(text)
        logging.info(text)

    def run_job(self, title, fn, callback):
        if self.worker is not None:
            return
        for action in self.actions:
            action.setEnabled(False)
        self.nominal.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.log(title)
        self.callback = callback
        self.worker = Worker(fn, self)
        self.worker.progress.connect(self.log)
        self.worker.result.connect(self.job_result)
        self.worker.error.connect(self.job_error)
        self.worker.finished.connect(self.job_finished)
        self.worker.start()

    def job_result(self, result):
        try:
            if self.callback:
                self.callback(result)
        except Exception as exc:
            self.job_error(str(exc))
            logging.exception('Erro ao apresentar resultados')

    def job_error(self, text):
        self.log(f'Não foi possível concluir: {text}')
        QMessageBox.warning(self, 'Operação não concluída', text)

    def job_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.callback = None
        self.progress.hide()
        for action in self.actions:
            action.setEnabled(True)
        self.nominal.setEnabled(True)
        self.update_model_controls()

    def refresh(self):
        self.run_job('Lendo histórico...', lambda log: analysis(self.store, log), self.display)

    def display(self, state):
        self.state = state
        s = state['summary']
        df = state['df']
        if not df.empty and not self.date_initialized:
            first = df.index.min().tz_convert(LOCAL_TZ).date()
            last = df.index.max().tz_convert(LOCAL_TZ).date()
            for edit in [self.start_date, self.end_date]:
                edit.setDateRange(QDate(first.year, first.month, first.day), QDate(last.year, last.month, last.day))
            self.end_date.setDate(QDate(last.year, last.month, last.day))
            self.start_date.setDate(self.end_date.date().addDays(-6))
            self.date_initialized = True
        elif not df.empty:
            first = df.index.min().tz_convert(LOCAL_TZ).date()
            last = df.index.max().tz_convert(LOCAL_TZ).date()
            for edit in [self.start_date, self.end_date]:
                edit.setDateRange(QDate(first.year, first.month, first.day), QDate(last.year, last.month, last.day))
        self.card_values[3].setText(f"{s['rows']:,}".replace(',', '.'))
        if s['rows']:
            self.data_note.setText(f"{s['start']} → {s['end']}  •  {s['rows']:,} registros únicos  •  {s['missing']} intervalos ausentes.\n"
                                   f"Códigos Validacao (sem interpretação): {s['validation']}")
        else:
            self.data_note.setText('Histórico vazio. Importe o arquivo da pasta examples ou outro CSV compatível.')
        rows = [[local_label(ts), f'{r.pdc:.2f}', f'{r.poa:.2f}', '' if pd.isna(r.validation) else r.validation] for ts,r in df.tail(500).iloc[::-1].iterrows()]
        fill_table(self.data_table, rows)
        fill_table(self.import_table, state['imports'].values.tolist())
        events = state['alerts']
        fill_table(self.alert_table, [[e['date'], e['type'], e['detail']] for e in events[:1000]])
        self.alert_note.setText(f'{len(events)} eventos no histórico. Exibindo até 1.000; a exportação inclui todos. '
                               + ('Desvios avaliados somente após o período de ajuste do modelo.' if state['model'] else 'Ative um modelo para avaliar desvios de geração.'))
        self.update_model_controls()
        self.plot()
        self.log('Histórico e análises atualizados. Tudo salvo neste computador.')

    def update_model_controls(self):
        version = self.store.setting('active_model')
        candidate = self.store.setting('candidate_model')
        info = load_info(self.store.home, candidate) if candidate else (load_info(self.store.home, version) if version else None)
        self.activate_button.setEnabled(bool(info and info['eligible'] and candidate != version and self.worker is None))
        self.rollback_button.setEnabled(bool(self.store.setting('previous_model') and self.worker is None))
        self.model_status.setText(f'Em uso: {version}' if version else 'Nenhum modelo ativo. Treine uma versão e confira sua avaliação antes de ativar.')
        if info:
            rows = []
            for name, score in [('Candidata' if candidate != version else 'Modelo em uso', info['metrics']), ('Persistência (última potência)', info['persistence']), ('Versão anterior no mesmo teste', info.get('incumbent'))]:
                if score:
                    rows.append([name, f"{score['mae']:.2f}", f"{score['rmse']:.2f}", f"{score['r2']:.4f}" if score['r2'] is not None else '—', score['n']])
            fill_table(self.model_table, rows)
            t = info['periods']['test']
            self.model_detail.setText(f"Teste: {local_label(t['start'])} a {local_label(t['end'])}.\n"
                                      f"Candidata {'aprovada pelo critério de MAE' if info['eligible'] else 'não aprovada pelo critério de MAE'}. "
                                      'Esta avaliação não valida diagnóstico de sujidade nem garante desempenho em novos períodos.')
        if self.state and self.state['model']:
            self.card_values[2].setText(f"{self.state['model']['metrics']['mae']:.1f} W")
        else:
            self.card_values[2].setText('Sem modelo')

    def plot(self):
        if self.state is None:
            return
        start = pd.Timestamp(self.start_date.date().toPython(), tz=LOCAL_TZ)
        end = pd.Timestamp(self.end_date.date().toPython(), tz=LOCAL_TZ) + pd.Timedelta(days=1)
        if start >= end:
            QMessageBox.information(self, 'Período inválido', 'A data inicial deve ser anterior ou igual à data final.')
            return
        df = self.state['df']
        selected = df[(df.index >= start) & (df.index < end)]
        self.card_values[0].setText(f'{selected.pdc.sum() * .25 / 1000:.2f} kWh')
        self.card_values[1].setText(f'{selected.pdc.max():.1f} W' if not selected.empty else '—')
        self.figure.clear()
        ax = self.figure.add_subplot(211)
        bx = self.figure.add_subplot(212, sharex=ax)
        pred = self.state['prediction'].reindex(selected.index)
        # Reamostragem apenas visual; cálculos/alertas continuam em 15 minutos.
        chart = pd.DataFrame({'real': selected.pdc, 'prev': pred, 'poa': selected.poa})
        if len(chart) > 2500:
            chart = chart.resample('3h').mean()
            resolution = 'Médias de 3 h na visualização'
        else:
            chart = chart.asfreq('15min') if not chart.empty else chart
            resolution = 'Resolução de 15 min'
        x = chart.index.tz_convert(LOCAL_TZ).to_pydatetime() if not chart.empty else []
        ax.plot(x, chart.real, color=GREEN, linewidth=1.5, label='Potência real')
        ax.fill_between(x, 0, chart.real, color=GREEN, alpha=.06)
        if chart.prev.notna().any():
            ax.plot(x, chart.prev, color='#b68730', linewidth=1.4, linestyle='--', label='Esperada • ensemble')
        bx.plot(x, chart.poa, color='#82964d', linewidth=1.2, label='Irradiância no plano dos módulos')
        bx.fill_between(x, 0, chart.poa, color='#cddda5', alpha=.35)
        ax.set_ylabel('Potência DC (W)', fontsize=9, color=MUTED)
        bx.set_ylabel('POA (W/m²)', fontsize=9, color=MUTED)
        for axis in [ax, bx]:
            axis.spines[['top', 'right']].set_visible(False)
            axis.spines[['left', 'bottom']].set_color('#e6ebe2')
            axis.grid(axis='y', color='#edf1e9', linewidth=.8)
            axis.tick_params(colors=MUTED, labelsize=8, length=0, pad=7)
            axis.set_ylim(bottom=0)
            axis.legend(loc='upper left', frameon=False, fontsize=8)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7, tz=LOCAL_TZ)
        bx.xaxis.set_major_locator(locator)
        bx.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M', tz=LOCAL_TZ))
        ax.tick_params(labelbottom=False)
        self.canvas.draw_idle()
        self.chart_note.setText(f'{resolution}  •  Horário UTC−03  •  ' + ('Previsões disponíveis no período selecionado.' if pred.notna().any() else 'Sem previsões neste período; o modelo só é exibido após seu ajuste.'))
        nominal = self.store.setting('nominal_w')
        if nominal and not selected.empty:
            day = selected[selected.poa > 50]
            pr = day.pdc.sum() / (nominal * day.poa.sum() / 1000) if day.poa.sum() > 0 else np.nan
            self.overview_note.setText(f'PR do período: {pr:.3f}  •  Referência nominal: {nominal:g} W. FF indisponível: parâmetros/temperatura não confirmados.')
        else:
            self.overview_note.setText('PR aguarda potência nominal confirmada. FF aguarda parâmetros e temperatura. Configure a referência em Configurações.')

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Importar medições', str(ROOT / 'examples'), 'CSV (*.csv)')
        if not paths:
            return
        def task(log):
            messages = []
            for path in paths:
                try:
                    log(f'Importando {Path(path).name}...')
                    r = self.store.import_csv(path)
                    message = f"{r['file']}: arquivo já importado." if r['unchanged'] else f"{r['file']}: {r['added']} novos; {r['duplicates']} repetidos."
                except Exception as exc:
                    logging.exception('Importação recusada')
                    message = f'{Path(path).name}: RECUSADO — {exc}'
                messages.append(message)
                log(message)
            return analysis(self.store, log), messages
        def done(result):
            self.date_initialized = False
            self.display(result[0])
            QMessageBox.information(self, 'Resultado da importação', '\n\n'.join(result[1]))
        self.run_job('Importação iniciada...', task, done)

    def train(self):
        self.navigate(2)
        self.run_job('Preparando uma nova versão...', lambda log: train_candidate(self.store, log), lambda info: self.update_model_controls())

    def activate(self):
        def task(log):
            activate_candidate(self.store)
            return analysis(self.store, log)
        self.run_job('Ativando candidata aprovada...', task, self.display)

    def restore(self):
        def task(log):
            rollback(self.store)
            return analysis(self.store, log)
        self.run_job('Restaurando versão anterior...', task, self.display)

    def save_settings(self):
        value = self.nominal.value() or None
        def task(log):
            self.store.set_setting('nominal_w', value)
            return analysis(self.store, log)
        self.run_job('Recalculando indicadores...', task, self.display)

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Exportar histórico', str(ROOT / 'historico_exportado.csv'), 'CSV (*.csv)')
        if not path:
            return
        def task(log):
            df = self.store.read().rename(columns={'poa':'POA','pdc':'Pdc1','vdc':'Vdc1','idc':'Idc1','validation':'Validacao','temp_air':'Tar med'})
            df = df.drop(columns=['source'])
            # Opcionais completamente vazios são omitidos para permitir reimportação.
            df.dropna(axis=1, how='all').to_csv(path, encoding='utf-8-sig')
            return path
        self.run_job('Exportando histórico...', task, lambda p: self.log(f'Histórico exportado: {p}'))

    def export_alerts(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Exportar alertas', str(ROOT / 'alertas.csv'), 'CSV (*.csv)')
        if path and self.state:
            rows = [{k:v for k,v in a.items() if k != 'ts'} for a in self.state['alerts']]
            self.run_job('Exportando alertas...', lambda log: pd.DataFrame(rows, columns=['date','type','detail']).to_csv(path, index=False, encoding='utf-8-sig'), lambda _: self.log(f'Alertas exportados: {path}'))

    def backup(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Backup do histórico', str(ROOT / 'historico_backup.sqlite3'), 'SQLite (*.sqlite3)')
        if path:
            self.run_job('Criando backup consistente...', lambda log: self.store.backup(path), lambda _: self.log(f'Backup do histórico criado: {path}'))

    def closeEvent(self, event):
        if self.worker is not None:
            QMessageBox.information(self, 'Operação em andamento', 'Aguarde a operação terminar antes de fechar. A janela permanece disponível durante o processamento.')
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    args = parser.parse_args()
    home = Path(args.data_dir)
    home.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(home/'solardesk.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8')], format='%(asctime)s %(levelname)s %(message)s')
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    lock = QLockFile(str(home / 'interface.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None, 'SolarDesk já aberto', 'Já existe uma janela usando este histórico. Volte à janela existente.')
        return 0
    window = Window(Store(home))
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
