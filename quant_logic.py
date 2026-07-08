import os
import re
import ast
import time
import logging
import operator
import zipfile
import sqlite3
import datetime as dt
from io import BytesIO
from xml.etree import ElementTree as ET

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

try:  # 공용 시세 레이어(pykrx/yfinance) — 없으면 기존 네이버 크롤만 사용
    import arcmarket
except ImportError:
    arcmarket = None

logger = logging.getLogger(__name__)

# 외부 HTTP 호출 기본 타임아웃(초). 응답이 없을 때 백테스트 스레드가
# 무한 대기하지 않도록 모든 requests 호출에 강제한다.
HTTP_TIMEOUT = 10


class SafeExprError(Exception):
    """safe_expr 가 허용하지 않는 노드/구문을 만났을 때."""


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Lt: operator.lt, ast.Gt: operator.gt, ast.LtE: operator.le,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}


def safe_expr(expr, names=None):
    """내장 평가 함수를 대체하는 ast 화이트리스트 평가기.

    숫자/불리언 리터럴, 비교(체이닝 포함), and/or/not, 사칙·거듭제곱,
    괄호, 단항 ±, names 로 주입한 식별자만 허용한다. 함수 호출·속성
    접근·구독·람다·import 등 그 외 노드는 SafeExprError 로 거부한다.
    허용 식에 대해서는 파이썬 표준 의미(단축평가 반환값 포함)와 동일.
    """
    names = names or {}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise SafeExprError(f"syntax error: {e}")
    return _safe_visit(tree.body, names)


def _safe_visit(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise SafeExprError(f"unknown name: {node.id}")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _safe_visit(v, names)
                if not result:
                    return result
            return result
        result = False
        for v in node.values:
            result = _safe_visit(v, names)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_visit(node.operand, names))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_visit(node.left, names),
                                       _safe_visit(node.right, names))
    if isinstance(node, ast.Compare):
        left = _safe_visit(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _CMP_OPS:
                raise SafeExprError(f"operator not allowed: {op_type.__name__}")
            right = _safe_visit(comparator, names)
            if not _CMP_OPS[op_type](left, right):
                return False
            left = right
        return True
    raise SafeExprError(f"node not allowed: {type(node).__name__}")


def price_on_or_before(df, date, col='Close'):
    """date 의 종가를 반환. 해당 일자가 없으면 그 이전 마지막 거래일의
    종가로 폴백하고, 데이터가 전혀 없으면 0 을 반환한다."""
    try:
        if date in df.index:
            return df.loc[date, col]
        prior = df[df.index < date]
        if not prior.empty:
            return prior.iloc[-1][col]
    except Exception:
        pass
    return 0

YEAR_MAPPING = {
    2024: 'D-1y_data',
    2023: 'D-2y_data',
    2022: 'D-3y_data',
    2021: 'D-4y_data',
    2020: 'D-5y_data'
}

METRICS_LIST = [
    'ROE', 'ROA', 'GPM', 'OPM', 'NPM', 'EBITDA',
    '부채비율', '유동비율', '당좌비율', '자기자본비율',
    '매출액증가율', '영업이익증가율', 'EPS증가율', 'BPS증가율',
    '재고자산회전율', '총자산회전율',
    'PER', 'PBR', 'PSR', 'PCR', 'EV/EBITDA', 'GP/A',
    'OCF', 'FCF'
]

METRIC_UNITS = {
    'ROE': '배', 'ROA': '배', 'GPM': '배', 'OPM': '배', 'NPM': '배',
    '부채비율': '%', '유동비율': '%', '당좌비율': '%', '자기자본비율': '%',
    '매출액증가율': '%', '영업이익증가율': '%', 'EPS증가율': '%', 'BPS증가율': '%',
    '재고자산회전율': '회', '총자산회전율': '회',
    'PER': '배', 'PBR': '배', 'PSR': '배', 'PCR': '배', 'EV/EBITDA': '배', 'GP/A': '배',
    'EBITDA': '억원', 'OCF': '억원', 'FCF': '억원'
}

TECH_INDICATORS_MAP = {
    '골든크로스': 'GC_5_20', '데드크로스': 'DC_5_20',
    '5일선': 'SMA_5', '20일선': 'SMA_20', '60일선': 'SMA_60', 
    '1년선': 'SMA_240', '3년선': 'SMA_720',
    '거래대금': 'TradingValue',
    'RSI': 'RSI', 'MACD': 'MACD',
    'STOCH_K': 'Stoch_K', 'STOCH_D': 'Stoch_D', 'OBV': 'OBV',
    '기관순매수': 'Inst_Net_Amt', '외국인순매수': 'Foreign_Net_Amt', '개인순매수': 'Personal_Net_Amt',
    '전일종가': 'Prev_Close_Chg', 
    '전일거래량': 'Prev_Volume_Chg',
    '전일거래대금': 'Prev_Value_Chg',
    '양봉': 'Is_Yangbong', '음봉': 'Is_Yinbong', 
    '장대양봉': 'Is_Big_Yang', '장대음봉': 'Is_Big_Yin',
    '십자캔들': 'Is_Doji', 
    '망치형': 'Is_Hammer', '역망치형': 'Is_Inv_Hammer',
    '유성형': 'Is_Shooting_Star', '교수형': 'Is_Hanging_Man',
    '비석형도지': 'Is_Gravestone', '잠자리도지': 'Is_Dragonfly',
    '상승장악형': 'Is_Bull_Engulf', '하락장악형': 'Is_Bear_Engulf',
    '관통형': 'Is_Piercing', '흑운형': 'Is_Dark_Cloud',
    '적삼병': 'Is_Red_Soldiers', '흑삼병': 'Is_Black_Crows',
    '유상증자': 'Event_RightIssue', '무상증자': 'Event_BonusIssue',
    '자사주취득': 'Event_TreasuryAcq', '자사주처분': 'Event_TreasuryDisp',
    '주식분할': 'Event_StockSplit', '감자': 'Event_CapitalReduction',
    '합병': 'Event_Merger', '타법인주식취득': 'Event_Acquisition',
    '단일판매': 'Event_Supply', '공급계약': 'Event_Supply', '전환사채': 'Event_CB',
    '신주인수권부사채': 'Event_BW', '교환사채': 'Event_EB', '배당': 'Event_Dividend',
    '유형자산': 'Event_AssetAcq', '영업양수도': 'Event_BizTransfer',
    '상한가': 'Is_UpperLimit', '하한가': 'Is_LowerLimit',
    '2일연속상한가': 'Is_2ConsecutiveUpper', '2일연속하한가': 'Is_2ConsecutiveLower',
    '3일연속상한가': 'Is_3ConsecutiveUpper', '3일연속하한가': 'Is_3ConsecutiveLower',
    '52주신고가': 'Is_52W_High', '52주신저가': 'Is_52W_Low',
    '볼린저밴드상단돌파': 'BB_Upper_Break', '볼린저밴드하단돌파': 'BB_Lower_Break'
}

class DBManager:
    def __init__(self):
        # Oracle Cloud (Linux) path handling (ensure absolute or relative safely)
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.dir_name = os.path.join(base_path, "Memories")
        os.makedirs(self.dir_name, exist_ok=True)
        
        self.db_path = os.path.join(self.dir_name, "stock_cache.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    code TEXT,
                    date TEXT,
                    open INTEGER, high INTEGER, low INTEGER, close INTEGER, volume INTEGER,
                    PRIMARY KEY (code, date)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS investor_data (
                    code TEXT,
                    date TEXT,
                    inst_net INTEGER, foreign_net INTEGER,
                    PRIMARY KEY (code, date)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS event_data (
                    code TEXT,
                    date TEXT,
                    event_col TEXT,
                    PRIMARY KEY (code, date, event_col)
                )
            """)

    def get_latest_date(self, code, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT MAX(date) FROM {table_name} WHERE code=?", (code,))
        res = cursor.fetchone()
        return res[0] if res and res[0] else None

    def _save_timeseries(self, code, df, table, value_cols):
        """code/date + 정수 값 컬럼을 INSERT OR IGNORE 로 저장한다.

        value_cols: [(db_col, df_col), ...] — DB 컬럼과 원본 DataFrame
        컬럼의 대응. save_price_data / save_investor_data 공통 골격.
        """
        if df.empty: return
        db_cols = ['code', 'date'] + [db for db, _ in value_cols]
        placeholders = ', '.join(['?'] * len(db_cols))
        sql = (f"INSERT OR IGNORE INTO {table} ({', '.join(db_cols)}) "
               f"VALUES ({placeholders})")
        data = []
        for index, row in df.iterrows():
            date_str = index.strftime('%Y-%m-%d') if isinstance(index, pd.Timestamp) else str(index)[:10]
            data.append((code, date_str, *[int(row[src]) for _, src in value_cols]))
        with self.conn:
            self.conn.executemany(sql, data)

    def save_price_data(self, code, df):
        self._save_timeseries(code, df, 'price_data', [
            ('open', 'Open'), ('high', 'High'), ('low', 'Low'),
            ('close', 'Close'), ('volume', 'Volume')])

    def save_investor_data(self, code, df):
        self._save_timeseries(code, df, 'investor_data', [
            ('inst_net', 'Inst_Net_Volume'), ('foreign_net', 'Foreign_Net_Volume')])

    def load_price_data(self, code):
        query = "SELECT date, open, high, low, close, volume FROM price_data WHERE code=? ORDER BY date ASC"
        df = pd.read_sql(query, self.conn, params=(code,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df

    def save_event_data(self, code, event_map):
        """event_map: {'2024-03-15': {'Event_BonusIssue': 1}, ...}"""
        rows = []
        for date_str, cols in event_map.items():
            for col, _ in cols.items():
                rows.append((code, date_str, col))
        if rows:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO event_data (code, date, event_col) VALUES (?, ?, ?)",
                    rows
                )

    def load_event_data(self, code):
        """Returns {'2024-03-15': {'Event_BonusIssue': 1}, ...} or None if no rows exist."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT date, event_col FROM event_data WHERE code=?", (code,))
        rows = cursor.fetchall()
        if not rows:
            return None
        result = {}
        for date_str, col in rows:
            result.setdefault(date_str, {})[col] = 1
        return result

    def has_event_cache(self, code):
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM event_data WHERE code=? LIMIT 1", (code,))
        return cursor.fetchone() is not None

    def load_investor_data(self, code):
        query = "SELECT date, inst_net, foreign_net FROM investor_data WHERE code=? ORDER BY date ASC"
        df = pd.read_sql(query, self.conn, params=(code,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.columns = ['Inst_Net_Volume', 'Foreign_Net_Volume']
        return df

class CrawlerUtil:
    @staticmethod
    def _arcmarket_daily(code, years, stop_date):
        """arcmarket(pykrx/yfinance) 일봉 → 레거시 규격(Date/Open/High/Low/Close/Volume, 오름차순).
        실패·미가용이면 None → 호출측이 네이버 크롤 폴백."""
        if arcmarket is None:
            return None
        try:
            df = arcmarket.kr_daily(code, days=int(years * 365))
        except Exception:
            return None
        if df is None or df.empty:
            return None
        out = df.reset_index().rename(columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"})
        for c in ("Open", "High", "Low", "Close", "Volume"):
            out[c] = out[c].fillna(0).astype(int)
        if stop_date is not None:
            out = out[out["Date"] > pd.to_datetime(stop_date)]
        return out.sort_values("Date").reset_index(drop=True)

    @staticmethod
    def fetch_kospi_index(years=10):
        if arcmarket is not None:
            try:
                df = arcmarket.kr_index_daily("KOSPI", days=int(years * 365))
                if df is not None and not df.empty:
                    out = df.rename(columns={"close": "Close"})[["Close"]]
                    out.index.name = "Date"
                    return out
            except Exception:
                pass
        result = []
        max_pages = int(years * 26) + 10 
        if max_pages < 20: max_pages = 20
        target_date_limit = datetime.today() - timedelta(days=years*365)
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            search_pages = min(max_pages, 400)
            for page in range(1, search_pages + 1):
                url = f"https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page={page}"
                res = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
                soup = BeautifulSoup(res.text, 'lxml')
                rows = soup.select('table.type_1 tr')
                valid_rows = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 2: continue
                    try:
                        date_text = cols[0].text.strip()
                        if not date_text or date_text == '.': continue 
                        date = pd.to_datetime(date_text)
                        if date < target_date_limit:
                            return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')
                        close_text = cols[1].text.strip().replace(',', '')
                        if not close_text: continue
                        close = float(close_text)
                        result.append({'Date': date, 'Close': close})
                        valid_rows += 1
                    except Exception: continue
                if valid_rows == 0 and page > 1: break
        except Exception: pass
        if not result: return pd.DataFrame()
        return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')

    @staticmethod
    def fetch_naver_stock_html(code, years=10, stop_date=None):
        via = CrawlerUtil._arcmarket_daily(code, years, stop_date)
        if via is not None and not via.empty:
            return via
        result = []
        max_pages = int(years * 26) + 10 
        target_date_limit = datetime.today() - timedelta(days=years*365)
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            search_pages = min(max_pages, 400) 
            for page in range(1, search_pages + 1):
                url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
                res = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
                soup = BeautifulSoup(res.text, 'lxml')
                rows = soup.select('table.type2 tr')
                valid_rows = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) != 7: continue
                    try:
                        date_text = cols[0].text.strip()
                        if not date_text: continue
                        date = pd.to_datetime(date_text)
                        if stop_date and date <= stop_date:
                            return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
                        if date < target_date_limit:
                            return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
                        close = int(cols[1].text.replace(',', ''))
                        open_ = int(cols[3].text.replace(',', ''))
                        high = int(cols[4].text.replace(',', ''))
                        low = int(cols[5].text.replace(',', ''))
                        volume = int(cols[6].text.replace(',', ''))
                        result.append({'Date': date, 'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume})
                        valid_rows += 1
                    except Exception: continue
                if valid_rows == 0 and page > 1: break
        except Exception: pass
        if not result: return pd.DataFrame()
        return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

    @staticmethod
    def fetch_investor_data(code, years=1, stop_date=None):
        MAX_PAGES = int(years * 26) + 5
        TIMEOUT_LIMIT = 5
        url_base = f"https://finance.naver.com/item/frgn.nhn?code={code}&page="
        result = []
        start_date_dt = datetime.today() - timedelta(days=years*365)
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            for page in range(1, MAX_PAGES + 1):
                url = url_base + str(page)
                try:
                    res = requests.get(url, headers=headers, timeout=TIMEOUT_LIMIT)
                    soup = BeautifulSoup(res.text, 'lxml')
                    rows = soup.select('table.type2 tr')
                    valid_on_page = 0
                    for row in rows:
                        cols = row.find_all('td')
                        if not cols or not cols[0].text.strip(): continue
                        if len(cols) < 7: continue
                        try:
                            date_str = cols[0].text.strip().replace('.', '-')
                            date = pd.to_datetime(date_str)
                            if stop_date and date <= stop_date:
                                df = pd.DataFrame(result)
                                if df.empty: return df
                                return df.sort_values('Date').reset_index(drop=True)
                            if date < start_date_dt:
                                df = pd.DataFrame(result)
                                if df.empty: return df
                                return df.sort_values('Date').reset_index(drop=True)
                            inst_text = cols[5].text.strip()
                            fore_text = cols[6].text.strip()
                            if not inst_text or not fore_text: continue
                            inst_net_vol = int(inst_text.replace(',', '').replace('+', ''))
                            foreign_net_vol = int(fore_text.replace(',', '').replace('+', ''))
                            result.append({'Date': date, 'Inst_Net_Volume': inst_net_vol, 'Foreign_Net_Volume': foreign_net_vol})
                            valid_on_page += 1
                        except Exception: continue
                    if valid_on_page == 0 and page > 5: break
                    next_page = soup.select_one('td.pgR a')
                    if not next_page:
                         if page > 10: break
                except Exception: continue
        except Exception: return pd.DataFrame()
        df = pd.DataFrame(result)
        if df.empty: return df
        return df.sort_values('Date').reset_index(drop=True)

class TechnicalAnalysis:
    # 가격제한폭(±30%)으로는 설명되지 않는 일중 갭 = 권리락(액면분할·무상증자·유상증자 등) 신호.
    # 시가가 전일 종가의 이 비율 미만으로 출발하면 권리락으로 간주한다. (0.70 = 약 -30% 이상 급락)
    CORP_ACTION_GAP_THRESHOLD = 0.70

    @staticmethod
    def adjust_for_corporate_actions(df):
        """
        네이버 일봉(sise_day)은 '원주가'(무보정) 시계열이라, 액면분할·무상증자·유상증자
        '권리락일'에 주가가 가치 변화 없이 산술적으로만 뚝 떨어진다. 백테스트는 이 갭을
        실손실로 오인하므로, 표준 '수정주가' 방식대로 권리락 '이전' 구간의 OHLC를 비율만큼
        소급 축소해 시계열을 연속적으로 만든다. (거래량은 역으로 확대 → 거래대금 = 가격×거래량 보존)

        df : index=Date, columns=['Open','High','Low','Close','Volume', ...]
        반환: 보정된 새 DataFrame (원본 미변경, DB에는 원주가 그대로 유지하고 메모리에서만 보정)
        """
        if df is None or df.empty or len(df) < 2:
            return df
        df = df.sort_index().copy()
        n = len(df)
        opens  = df['Open'].to_numpy(dtype=float)
        closes = df['Close'].to_numpy(dtype=float)

        # cf[i] = i행 가격에 곱할 누적 보정계수. 최근일=1.0, 권리락을 지날 때마다 과거 쪽이 작아진다.
        cf = np.ones(n)
        running = 1.0
        for i in range(n - 1, 0, -1):
            cf[i] = running
            # 상장 첫날(데이터 첫 행=index 0) 직후의 갭은 IPO 특유의 큰 변동성일 수 있으므로
            # 권리락 후보에서 제외한다. (즉 index 0↔1 전이는 절대 권리락으로 보지 않음)
            if i == 1:
                continue
            ratio = TechnicalAnalysis._corp_action_ratio(
                prev_close=closes[i - 1],
                today_open=opens[i],
                today_close=closes[i],
            )
            if ratio is not None and 0 < ratio < 50:   # 방어적 클램프 (역분할/감자면 1보다 클 수 있음)
                running *= ratio
        cf[0] = running

        for col in ('Open', 'High', 'Low', 'Close'):
            if col in df.columns:
                df[col] = df[col].to_numpy(dtype=float) * cf
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].to_numpy(dtype=float) / np.where(cf == 0, 1.0, cf)
        return df

    @staticmethod
    def _corp_action_ratio(prev_close, today_open, today_close):
        """
        전일 종가 대비 당일 '시가' 갭이 권리락(액면분할·무상증자·유상증자 / 역분할·감자)에 의한
        것인지 판정하고, 맞으면 '권리락 이전 가격에 곱할 보정 비율'을 돌려준다. 아니면 None.

        정책 (사용자 결정):
          · 기준값 = today_open / prev_close  (시가 기준 — 당일 등락 노이즈 배제)
          · 임계치 = CORP_ACTION_GAP_THRESHOLD (=0.70)
              - 하방: gap < 0.70   → 정상 하한가(-30%)로도 설명 불가 → 분할·무상·유상증자 (비율 < 1)
              - 상방: gap > 1/0.70 → 정상 상한가(+30%)로도 설명 불가 → 액면병합·감자       (비율 > 1)
          · 소규모 유상증자(-5~20%)·소규모 감자는 진짜 급등락과 구분 불가 → 의도적으로 미보정
          · 상장 첫날의 IPO 변동성은 호출부(adjust_for_corporate_actions)에서 별도로 제외함
        """
        if prev_close <= 0 or today_open <= 0:
            return None
        gap = today_open / prev_close
        th = TechnicalAnalysis.CORP_ACTION_GAP_THRESHOLD
        if gap < th or gap > 1.0 / th:
            return gap
        return None

    @staticmethod
    def add_indicators(df):
        if df.empty: return df
        df['TradingValue'] = (df['Close'] * df['Volume']) / 100000000
        for window in [5, 20, 60, 120, 240, 720]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        prev_sma5 = df['SMA_5'].shift(1)
        prev_sma20 = df['SMA_20'].shift(1)
        df['GC_5_20'] = (prev_sma5 < prev_sma20) & (df['SMA_5'] > df['SMA_20'])
        df['DC_5_20'] = (prev_sma5 > prev_sma20) & (df['SMA_5'] < df['SMA_20'])
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        denom = (high_max - low_min).replace(0, np.nan)
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / denom)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Prev_Close_Chg'] = df['Close'].pct_change().shift(1) * 100
        df['Prev_Volume_Chg'] = df['Volume'].pct_change().shift(1) * 100
        df['Prev_Value_Chg'] = df['TradingValue'].pct_change().shift(1) * 100
        if 'Inst_Net_Volume' in df.columns and 'Foreign_Net_Volume' in df.columns:
            df['Inst_Net_Amt'] = (df['Inst_Net_Volume'] * df['Close']) / 100000000
            df['Foreign_Net_Amt'] = (df['Foreign_Net_Volume'] * df['Close']) / 100000000
            personal_vol = -(df['Inst_Net_Volume'] + df['Foreign_Net_Volume'])
            df['Personal_Net_Amt'] = (personal_vol * df['Close']) / 100000000
        else:
            df['Inst_Net_Amt'] = 0
            df['Foreign_Net_Amt'] = 0
            df['Personal_Net_Amt'] = 0

        O = df['Open']
        C = df['Close']
        H = df['High']
        L = df['Low']
        body = np.abs(C - O)
        upper_shadow = H - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - L

        df['Is_Yangbong'] = C > O
        df['Is_Yinbong'] = C < O
        body_pct = (body / O) * 100
        df['Is_Big_Yang'] = (df['Is_Yangbong']) & (body_pct >= 5.0)
        df['Is_Big_Yin'] = (df['Is_Yinbong']) & (body_pct >= 5.0)
        
        df['Is_Doji'] = body <= (O * 0.003)
        df['Is_Gravestone'] = df['Is_Doji'] & (lower_shadow <= body) & (upper_shadow > body * 2)
        df['Is_Dragonfly'] = df['Is_Doji'] & (upper_shadow <= body) & (lower_shadow > body * 2)
        
        df['Is_Hammer'] = (lower_shadow > body * 2) & (upper_shadow < body * 0.5)
        df['Is_Inv_Hammer'] = (upper_shadow > body * 2) & (lower_shadow < body * 0.5)
        df['Is_Shooting_Star'] = df['Is_Inv_Hammer']
        df['Is_Hanging_Man'] = df['Is_Hammer']
        
        prev_C = C.shift(1)
        prev_O = O.shift(1)
        prev_Yang = df['Is_Yangbong'].shift(1)
        prev_Yin = df['Is_Yinbong'].shift(1)

        df['Is_Bull_Engulf'] = prev_Yin & df['Is_Yangbong'] & (C > prev_O) & (O < prev_C)
        df['Is_Bear_Engulf'] = prev_Yang & df['Is_Yinbong'] & (C < prev_O) & (O > prev_C)
        mid_point = (prev_O + prev_C) / 2
        df['Is_Piercing'] = (df['Is_Big_Yin'].shift(1)) & df['Is_Yangbong'] & (O < prev_C) & (C > mid_point)
        df['Is_Dark_Cloud'] = (df['Is_Big_Yang'].shift(1)) & df['Is_Yinbong'] & (O > prev_C) & (C < mid_point)

        df['Is_Red_Soldiers'] = (
            df['Is_Yangbong'] & df['Is_Yangbong'].shift(1) & df['Is_Yangbong'].shift(2) &
            (C > C.shift(1)) & (C.shift(1) > C.shift(2))
        )
        df['Is_Black_Crows'] = (
            df['Is_Yinbong'] & df['Is_Yinbong'].shift(1) & df['Is_Yinbong'].shift(2) &
            (C < C.shift(1)) & (C.shift(1) < C.shift(2))
        )
        
        # 상한가 / 하한가 (대략 29.5% 이상, -29.5% 이하)
        pct_chg = df['Close'].pct_change() * 100
        df['Is_UpperLimit'] = pct_chg >= 29.5
        df['Is_LowerLimit'] = pct_chg <= -29.5
        df['Is_2ConsecutiveUpper'] = df['Is_UpperLimit'] & df['Is_UpperLimit'].shift(1)
        df['Is_2ConsecutiveLower'] = df['Is_LowerLimit'] & df['Is_LowerLimit'].shift(1)
        df['Is_3ConsecutiveUpper'] = df['Is_UpperLimit'] & df['Is_UpperLimit'].shift(1) & df['Is_UpperLimit'].shift(2)
        df['Is_3ConsecutiveLower'] = df['Is_LowerLimit'] & df['Is_LowerLimit'].shift(1) & df['Is_LowerLimit'].shift(2)
        
        # 52주 신고가/신저가 (약 252일)
        df['Is_52W_High'] = df['Close'] >= df['High'].rolling(252, min_periods=1).max()
        df['Is_52W_Low'] = df['Close'] <= df['Low'].rolling(252, min_periods=1).min()

        # 볼린저 밴드
        std_20 = df['Close'].rolling(window=20).std()
        bb_upper = df['SMA_20'] + (std_20 * 2)
        bb_lower = df['SMA_20'] - (std_20 * 2)
        prev_close = df['Close'].shift(1)
        prev_bb_upper = bb_upper.shift(1)
        prev_bb_lower = bb_lower.shift(1)
        df['BB_Upper_Break'] = (prev_close < prev_bb_upper) & (df['Close'] > bb_upper)
        df['BB_Lower_Break'] = (prev_close > prev_bb_lower) & (df['Close'] < bb_lower)

        # Event columns — populated later from DART; default 0
        for ev in ['RightIssue', 'BonusIssue', 'TreasuryAcq', 'TreasuryDisp', 'StockSplit', 
                   'CapitalReduction', 'Merger', 'Acquisition', 'Supply', 'CB', 'BW', 'EB', 
                   'Dividend', 'AssetAcq', 'BizTransfer']:
            df[f'Event_{ev}'] = 0

        return df

class QuantLogic:
    def __init__(self):
        self.db_manager = DBManager()
        self.financial_data = {}
        self.load_csv_files()

    def clean_financial_data(self, df):
        pct_cols = [k for k, v in METRIC_UNITS.items() if v == '%']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                non_zero = df[df[col] != 0][col].abs()
                if not non_zero.empty:
                    mean_val = non_zero.mean()
                    if mean_val < 1.0: 
                        df[col] = df[col] * 100
        won_to_eok_cols = [k for k, v in METRIC_UNITS.items() if v == '억원']
        for col in won_to_eok_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                non_zero = df[df[col] != 0][col].abs()
                if not non_zero.empty:
                    mean_val = non_zero.mean()
                    if mean_val > 1000000:
                        df[col] = df[col] / 100000000
        return df

    def load_csv_files(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        for year, key_name in YEAR_MAPPING.items():
            file_name = f"{key_name}.csv"
            file_path = os.path.join(base_path, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                except Exception:
                    df = pd.read_csv(file_path, encoding='cp949')
                if '종목코드' in df.columns:
                    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
                df = self.clean_financial_data(df)
                self.financial_data[key_name] = df.to_dict('records')

    def _extract_dart_value(self, grp, keywords, col='amt'):
        """DART 재무제표 그룹에서 계정과목 키워드로 값을 추출"""
        for kw in keywords:
            mask = grp['account_nm'].str.strip() == kw
            if mask.any():
                try: return float(str(grp.loc[mask, col].iloc[0]).replace(',', ''))
                except Exception: pass
            mask = grp['account_nm'].str.contains(kw, na=False, regex=False)
            if mask.any():
                try: return float(str(grp.loc[mask, col].iloc[0]).replace(',', ''))
                except Exception: pass
        return 0.0

    def _fetch_naver_market_data(self, progress_callback=None):
        """
        네이버 금융 sise_market_sum 페이지에서 전 종목 주가/시가총액/PER 일괄 크롤링
        컬럼 순서: N, 종목명(link), 현재가, 전일비, 등락률, 액면가, 시가총액(억), 상장주식수,
                   외국인비율, 거래량, PER, ROE, 토론
        Returns: {종목코드: {'주가': int, '시가총액': int(원), 'PER': float}}
        """
        hdrs   = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                  'Referer': 'https://finance.naver.com'}
        result = {}

        def _clean(s):
            return s.replace(',', '').replace('+', '').replace(' ', '').replace('N/A', '0') \
                    .replace('하락', '').replace('상승', '').replace('보합', '0') \
                    .replace('▼', '').replace('▲', '').strip()

        for sosok, mkt_name in [(0, 'KOSPI'), (1, 'KOSDAQ')]:
            if progress_callback:
                progress_callback(None, f"  네이버 금융 {mkt_name} 주가/시가총액 수집 중...")
            page = 1
            while page <= 200:
                try:
                    url  = (f"https://finance.naver.com/sise/"
                            f"sise_market_sum.naver?sosok={sosok}&page={page}")
                    res  = requests.get(url, headers=hdrs, timeout=10)
                    if res.status_code != 200:
                        break
                    soup = BeautifulSoup(res.text, 'html.parser')
                    rows = soup.select('table.type_2 tr')

                    valid = 0
                    for row in rows:
                        link = row.select_one('a[href*="code="]')
                        if not link:
                            continue
                        tds = row.find_all('td')
                        if len(tds) < 11:
                            continue
                        try:
                            code   = link['href'].split('code=')[-1].strip().zfill(6)
                            price  = int(_clean(tds[2].get_text()))
                            mc_str = _clean(tds[6].get_text())
                            mktcap = int(float(mc_str) * 1e8) if mc_str and mc_str != '0' else 0
                            per_s  = _clean(tds[10].get_text())
                            per    = float(per_s) if per_s and per_s not in ('0','') else 0.0
                            if code and price > 0:
                                result[code] = {'주가': price, '시가총액': mktcap, 'PER': per}
                                valid += 1
                        except Exception:
                            continue

                    if valid == 0:
                        break
                    page += 1
                    time.sleep(0.1)
                except Exception:
                    break

        if progress_callback:
            progress_callback(None, f"  네이버 주가 수집 완료: {len(result):,}개 종목")
        return result

    def _get_year_end_price(self, code, year):
        """연도 연말 종가 반환 (12/31 → 12/30 → 12/29 → 12/28 순서).
        SQLite 캐시 우선 조회, 없으면 네이버 sise_day 크롤링."""
        candidates = []
        for day in [31, 30, 29, 28]:
            try:
                candidates.append(dt.date(year, 12, day))
            except ValueError:
                pass

        # 1. SQLite 캐시 조회
        try:
            price_df = self.db_manager.load_price_data(code)
            if price_df is not None and not price_df.empty:
                idx_strs = price_df.index.strftime('%Y-%m-%d').tolist()
                close_col = 'close' if 'close' in price_df.columns else price_df.columns[0]
                for cand in candidates:
                    s = cand.strftime('%Y-%m-%d')
                    if s in idx_strs:
                        val = price_df.loc[price_df.index.strftime('%Y-%m-%d') == s, close_col].iloc[0]
                        if val and val > 0:
                            return int(val)
        except Exception:
            pass

        # 2. 네이버 sise_day.nhn 크롤링
        headers = {'User-Agent': 'Mozilla/5.0'}
        today = dt.date.today()
        target_ref = dt.date(year, 12, 31)
        biz_days = max(1, (today - target_ref).days * 252 // 365)
        page_est  = max(1, biz_days // 10 - 2)

        # 탐색 범위: page_est -2 ~ page_est +3
        pages_to_try = list(range(max(1, page_est - 2), page_est + 4))

        fetched_rows = {}   # date_str -> close_price
        for page in pages_to_try:
            try:
                url  = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code != 200:
                    break
                soup = BeautifulSoup(resp.text, 'html.parser')
                rows = soup.select('table.type2 tr')
                for row in rows:
                    tds = row.find_all('td')
                    if len(tds) < 2:
                        continue
                    date_text  = tds[0].get_text(strip=True)  # e.g. "2023.12.29"
                    close_text = tds[1].get_text(strip=True).replace(',', '')
                    if not date_text or '.' not in date_text:
                        continue
                    try:
                        d = dt.datetime.strptime(date_text, '%Y.%m.%d').date()
                        fetched_rows[d] = int(close_text)
                    except Exception:
                        continue
            except Exception:
                continue

        for cand in candidates:
            if cand in fetched_rows and fetched_rows[cand] > 0:
                return fetched_rows[cand]

        return 0

    def _compute_metrics_from_dart(self, fin_df, market_data=None, progress_callback=None):
        """DART finstate_all 결과 DataFrame → KRX Quant Simulator CSV 형식으로 변환"""
        if 'stock_code' not in fin_df.columns:
            return pd.DataFrame()

        fin_df = fin_df.copy()
        fin_df = fin_df[fin_df['stock_code'].notna() & (fin_df['stock_code'].astype(str).str.strip() != '')]
        fin_df['stock_code'] = fin_df['stock_code'].astype(str).str.zfill(6)

        def to_float(v):
            try: return float(str(v).replace(',', '').strip())
            except Exception: return 0.0

        fin_df['amt']      = fin_df['thstrm_amount'].apply(to_float)
        fin_df['prev_amt'] = fin_df['frmtrm_amount'].apply(to_float) if 'frmtrm_amount' in fin_df.columns else 0.0

        BS = {
            '자산총계':   ['자산총계'],
            '부채총계':   ['부채총계'],
            '자본총계':   ['자본총계', '자본합계'],
            '유동자산':   ['유동자산'],
            '유동부채':   ['유동부채'],
            '재고자산':   ['재고자산'],
            '당좌자산':   ['당좌자산'],
        }
        IS = {
            '매출액':     ['매출액', '영업수익', '수익(매출액)', '매출'],
            '매출원가':   ['매출원가'],
            '매출총이익': ['매출총이익'],
            '영업이익':   ['영업이익', '영업손익'],
            '당기순이익': ['당기순이익', '당기순손익', '당기순이익(손실)'],
            'EPS':        ['기본주당이익(손실)', '기본주당순이익(손실)', '주당이익', '주당순이익'],
            'BPS':        ['주당순자산가치', '주당순자산'],
        }
        CF = {
            'OCF':   ['영업활동으로 인한 현금흐름', '영업활동현금흐름', '영업활동으로인한현금흐름'],
            'CAPEX': ['유형자산의 취득', '유형자산취득', '유형자산 취득'],
        }

        def pct(a, b):    return round(a/b*100, 2) if b else 0.0
        def ratio(a, b):  return round(a/b, 4)     if b else 0.0
        def growth(c, p): return round((c-p)/abs(p)*100, 2) if p else 0.0

        results = []
        groups = fin_df.groupby('stock_code')
        total  = len(groups)

        for idx, (code, grp) in enumerate(groups):
            if idx % 300 == 0 and progress_callback:
                progress_callback(None, f"  재무비율 계산 중... {idx}/{total}")
            try:
                corp = grp['corp_name'].iloc[0] if 'corp_name' in grp.columns else code

                tot_assets  = self._extract_dart_value(grp, BS['자산총계'])
                tot_liab    = self._extract_dart_value(grp, BS['부채총계'])
                tot_equity  = self._extract_dart_value(grp, BS['자본총계'])
                cur_assets  = self._extract_dart_value(grp, BS['유동자산'])
                cur_liab    = self._extract_dart_value(grp, BS['유동부채'])
                inventory   = self._extract_dart_value(grp, BS['재고자산'])
                quick_as    = self._extract_dart_value(grp, BS['당좌자산'])

                revenue     = self._extract_dart_value(grp, IS['매출액'])
                cogs        = self._extract_dart_value(grp, IS['매출원가'])
                gross_p     = self._extract_dart_value(grp, IS['매출총이익'])
                op_income   = self._extract_dart_value(grp, IS['영업이익'])
                net_income  = self._extract_dart_value(grp, IS['당기순이익'])
                eps         = self._extract_dart_value(grp, IS['EPS'])
                bps         = self._extract_dart_value(grp, IS['BPS'])

                prev_rev    = self._extract_dart_value(grp, IS['매출액'],   'prev_amt')
                prev_op     = self._extract_dart_value(grp, IS['영업이익'], 'prev_amt')
                prev_eps    = self._extract_dart_value(grp, IS['EPS'],      'prev_amt')
                prev_bps    = self._extract_dart_value(grp, IS['BPS'],      'prev_amt')

                ocf         = self._extract_dart_value(grp, CF['OCF'])
                capex       = self._extract_dart_value(grp, CF['CAPEX'])

                if gross_p == 0 and revenue and cogs:
                    gross_p = revenue - cogs
                quick_eff = quick_as if quick_as else max(0, cur_assets - inventory)

                # ── 네이버 시장 데이터 조회 ───────────────────────────────
                mkt       = (market_data or {}).get(code, {})
                price     = mkt.get('주가', 0)
                mktcap    = mkt.get('시가총액', 0)   # 원 단위
                per_naver = mkt.get('PER', 0.0)

                # PER: 네이버 제공값 우선, 없으면 주가/EPS 직접 계산
                if per_naver and per_naver > 0:
                    per_val = per_naver
                elif price and eps and eps > 0:
                    per_val = round(price / eps, 2)
                else:
                    per_val = 0.0

                # PBR = 주가 / BPS  (BPS: DART 원단위)
                pbr_val = round(price / bps, 2) if (price and bps and bps > 0) else 0.0

                # PSR = 시가총액 / 매출액  (모두 원 단위)
                psr_val = round(mktcap / revenue, 4) if (mktcap and revenue and revenue > 0) else 0.0

                # PCR = 시가총액 / 영업현금흐름
                pcr_val = round(mktcap / ocf, 4) if (mktcap and ocf and ocf > 0) else 0.0

                # EV/EBITDA ≈ (시가총액 + 부채총계) / 영업이익
                ev       = mktcap + tot_liab if mktcap else 0
                ev_ebitda_val = round(ev / op_income, 2) if (ev and op_income and op_income > 0) else 0.0

                results.append({
                    '종목코드': code, '기업명': corp,
                    'ROE': pct(net_income, tot_equity),
                    'ROA': pct(net_income, tot_assets),
                    'GPM': pct(gross_p, revenue),
                    'OPM': pct(op_income, revenue),
                    'NPM': pct(net_income, revenue),
                    'EBITDA': round(op_income / 1e8, 2),
                    '부채비율':     pct(tot_liab, tot_equity),
                    '유동비율':     pct(cur_assets, cur_liab),
                    '당좌비율':     pct(quick_eff, cur_liab),
                    '자기자본비율': pct(tot_equity, tot_assets),
                    '매출액증가율':   growth(revenue,   prev_rev),
                    '영업이익증가율': growth(op_income, prev_op),
                    'EPS증가율':    growth(eps, prev_eps),
                    'BPS증가율':    growth(bps, prev_bps),
                    '재고자산회전율': ratio(revenue, inventory if inventory else 1),
                    '총자산회전율':   ratio(revenue, tot_assets),
                    'PER': per_val, 'PBR': pbr_val,
                    'PSR': psr_val, 'PCR': pcr_val,
                    'EV/EBITDA': ev_ebitda_val,
                    'GP/A': ratio(gross_p, tot_assets),
                    'OCF': round(ocf / 1e8, 2),
                    'FCF': round((ocf - abs(capex)) / 1e8, 2) if capex else round(ocf / 1e8, 2),
                    '주가': price, '시가총액': mktcap,
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def update_financial_data(self, dart_key, progress_callback=None):
        """
        DART API로 최신 재무제표 확인 및 실제 업데이트 수행
        1단계: corpCode.xml로 전체 상장사 목록 수집
        2단계: fnlttMultiAcnt로 100개씩 배치 조회 (전체 ~40배치/년)
        3단계: 네이버 현재가 + 연말주가로 시장 지표 계산 후 CSV 저장
        """
        global YEAR_MAPPING

        DART_BASE = 'https://opendart.fss.or.kr/api'

        def dart_get(endpoint, params, timeout=30):
            r = requests.get(f'{DART_BASE}/{endpoint}', params={'crtfc_key': dart_key, **params}, timeout=timeout)
            return r.json()

        # ── 1. API 연결 테스트 (삼성전자 단일 조회) ────────────────────────
        if progress_callback: progress_callback(2, "DART API 연결 확인 중...")
        try:
            probe = dart_get('fnlttSinglAcnt.json',
                             {'corp_code': '00126380', 'bsns_year': datetime.now().year - 1,
                              'reprt_code': '11011'})
            if probe.get('status') not in ('000', '013'):
                msg = probe.get('message', '알 수 없는 오류')
                if progress_callback: progress_callback(None, f"DART API 오류: {msg}", msg)
                return
        except Exception as e:
            if progress_callback: progress_callback(None, f"DART 연결 실패: {e}", str(e))
            return

        # ── 2. 전체 상장사 corp_code 목록 수집 ────────────────────────────
        if progress_callback: progress_callback(5, "전체 상장사 corp_code 목록 수집 중...")
        try:
            resp = requests.get(f'{DART_BASE}/corpCode.xml?crtfc_key={dart_key}', timeout=30)
            z    = zipfile.ZipFile(BytesIO(resp.content))
            tree = ET.fromstring(z.read('CORPCODE.xml'))
            corps    = []   # [{corp_code, stock_code, corp_name}]
            name_map = {}   # stock_code -> corp_name
            for item in tree.findall('list'):
                sc = item.findtext('stock_code', '').strip()
                cc = item.findtext('corp_code',  '').strip()
                nm = item.findtext('corp_name',  '').strip()
                if sc and cc:
                    corps.append({'corp_code': cc, 'stock_code': sc, 'corp_name': nm})
                    name_map[sc] = nm
            if progress_callback:
                progress_callback(8, f"상장사 {len(corps):,}개 목록 수집 완료")
        except Exception as e:
            if progress_callback: progress_callback(None, f"corp_code 목록 오류: {e}", str(e))
            return

        # ── 3. DART 최신 사업연도 탐색 ────────────────────────────────────
        if progress_callback: progress_callback(10, "DART 최신 사업연도 탐색 중...")
        current_csv_year = max(YEAR_MAPPING.keys())
        latest_dart_year = current_csv_year
        for test_year in range(datetime.now().year - 1, current_csv_year - 2, -1):
            try:
                p = dart_get('fnlttSinglAcnt.json',
                             {'corp_code': '00126380', 'bsns_year': test_year, 'reprt_code': '11011'})
                if p.get('status') == '000' and p.get('list'):
                    latest_dart_year = test_year
                    break
            except Exception:
                continue
        if progress_callback:
            progress_callback(12, f"CSV 기준연도: {current_csv_year}년 | DART 최신연도: {latest_dart_year}년")

        # ── 4. 최신 여부 판단 ──────────────────────────────────────────────
        if latest_dart_year <= current_csv_year:
            d_1y    = self.financial_data.get('D-1y_data', [])
            samples = [x for x in d_1y[:10] if x.get('종목코드') and x.get('ROE')]
            is_outdated    = not samples   # 데이터 없으면 바로 업데이트
            verified_count = 0

            if not is_outdated:
                if progress_callback: progress_callback(14, f"샘플 ROE 대조 중...")
                for comp in samples[:3]:
                    code    = comp['종목코드']
                    csv_roe = float(comp.get('ROE', 0))
                    # corp_code 역조회
                    cc = next((c['corp_code'] for c in corps if c['stock_code'] == code), None)
                    if not cc: continue
                    try:
                        fin = dart_get('fnlttSinglAcnt.json',
                                       {'corp_code': cc, 'bsns_year': current_csv_year,
                                        'reprt_code': '11011'})
                        if fin.get('status') != '000' or not fin.get('list'): continue
                        df_fin = pd.DataFrame(fin['list'])
                        ni_r = df_fin[df_fin['account_nm'].str.contains('당기순이익', na=False)]
                        eq_r = df_fin[df_fin['account_nm'].str.contains('자본총계',  na=False)]
                        if ni_r.empty or eq_r.empty: continue
                        ni = float(str(ni_r.iloc[0]['thstrm_amount']).replace(',',''))
                        eq = float(str(eq_r.iloc[0]['thstrm_amount']).replace(',',''))
                        if eq == 0: continue
                        dart_roe = round(ni / eq * 100, 2)
                        verified_count += 1
                        if abs(dart_roe - csv_roe) > 1.5:
                            is_outdated = True
                            if progress_callback:
                                progress_callback(15, f"불일치: {comp.get('기업명', code)} "
                                                      f"ROE CSV={csv_roe:.2f}% / DART={dart_roe:.2f}%")
                            break
                        else:
                            if progress_callback:
                                progress_callback(None, f"일치: {comp.get('기업명', code)} "
                                                        f"ROE={csv_roe:.2f}% ≈ {dart_roe:.2f}%")
                    except Exception:
                        continue

                if not is_outdated and verified_count == 0:
                    is_outdated = True
                    if progress_callback:
                        progress_callback(15, "ROE 검증 불가 → 업데이트 진행")

            if not is_outdated:
                if progress_callback:
                    progress_callback(100, f"최신 재무제표입니다. ({current_csv_year}년 기준, "
                                           f"ROE {verified_count}개 종목 일치)")
                return

        # ── 5. 네이버 현재 주가/시가총액/PER 수집 ─────────────────────────
        update_year = latest_dart_year
        if progress_callback:
            progress_callback(18, f"{update_year}년 기준 업데이트 시작. 네이버 주가 수집 중...")
        market_data = self._fetch_naver_market_data(progress_callback)

        # ── 6. 연도별 배치 조회 ────────────────────────────────────────────
        base_path      = os.path.dirname(os.path.abspath(__file__))
        new_year_order = [(update_year - i, f'D-{i+1}y_data') for i in range(5)]
        corp_codes     = [c['corp_code'] for c in corps]
        BATCH          = 100

        for yr_idx, (year, key_name) in enumerate(new_year_order):
            base_pct = 22 + yr_idx * 14
            if progress_callback:
                progress_callback(base_pct,
                    f"[{yr_idx+1}/5] {year}년 전체 상장사 재무제표 수집 중 (DART fnlttMultiAcnt)...")
            try:
                all_rows = []
                batches  = [corp_codes[i:i+BATCH] for i in range(0, len(corp_codes), BATCH)]
                for b_idx, batch in enumerate(batches):
                    if b_idx % 10 == 0 and progress_callback:
                        progress_callback(None,
                            f"  배치 {b_idx+1}/{len(batches)} ({b_idx*BATCH}/{len(corp_codes)}개)")
                    try:
                        jo = dart_get('fnlttMultiAcnt.json',
                                      {'corp_code': ','.join(batch),
                                       'bsns_year': year, 'reprt_code': '11011'})
                        if jo.get('status') == '000' and jo.get('list'):
                            all_rows.extend(jo['list'])
                    except Exception:
                        continue
                    time.sleep(0.05)

                if not all_rows:
                    if progress_callback: progress_callback(None, f"{year}년 데이터 없음 – 스킵")
                    continue

                fin_df = pd.DataFrame(all_rows)
                # corp_name 추가
                fin_df['corp_name'] = fin_df['stock_code'].map(name_map).fillna('')
                # CFS 우선 (연결재무제표); CFS 없는 종목만 OFS 사용
                if 'fs_div' in fin_df.columns:
                    cfs_codes = set(fin_df[fin_df['fs_div'] == 'CFS']['stock_code'])
                    fin_df = fin_df[
                        (fin_df['fs_div'] == 'CFS') |
                        ((fin_df['fs_div'] == 'OFS') & ~fin_df['stock_code'].isin(cfs_codes))
                    ]

                if progress_callback:
                    progress_callback(base_pct + 7,
                        f"{year}년 {len(fin_df):,}행 수신. 재무비율 계산 중...")

                # 시장 지표: D-1y=현재가, D-2y~=연말주가
                if yr_idx == 0:
                    year_market = market_data
                else:
                    if progress_callback:
                        progress_callback(None, f"{year}년 연말 주가 조회 중...")
                    year_market = {}
                    all_sc = (fin_df['stock_code'].dropna().astype(str)
                                     .str.zfill(6).unique().tolist()
                              if 'stock_code' in fin_df.columns else [])
                    for c_idx, c in enumerate(all_sc):
                        if c_idx % 300 == 0 and progress_callback:
                            progress_callback(None, f"  연말주가 조회: {c_idx}/{len(all_sc)}")
                        cur       = market_data.get(c, {})
                        cur_p     = cur.get('주가', 0)
                        cur_mc    = cur.get('시가총액', 0)
                        yep = self._get_year_end_price(c, year)
                        if yep > 0:
                            approx_mc = int(cur_mc * yep / cur_p) if cur_p > 0 else 0
                            year_market[c] = {'주가': yep, '시가총액': approx_mc, 'PER': 0.0}

                result_df = self._compute_metrics_from_dart(fin_df, year_market, progress_callback)

                if result_df is not None and not result_df.empty:
                    fp = os.path.join(base_path, f"{key_name}.csv")
                    result_df.to_csv(fp, index=False, encoding='utf-8-sig')
                    result_df = self.clean_financial_data(result_df)
                    self.financial_data[key_name] = result_df.to_dict('records')
                    if progress_callback:
                        progress_callback(base_pct + 13,
                            f"✓ {year}년 완료: {len(result_df)}개 종목 → {key_name}.csv")
            except Exception as e:
                if progress_callback: progress_callback(None, f"{year}년 처리 오류: {e}")

        # ── 7. YEAR_MAPPING 갱신 ──────────────────────────────────────────
        YEAR_MAPPING = dict(new_year_order)
        if progress_callback:
            progress_callback(100, f"재무제표 업데이트 완료! ({update_year}년 기준 D-1y~D-5y 갱신)")

    def evaluate_financial_query(self, code, query):
        if not query.strip(): return True
        query = query.replace(">=", ">").replace("<=", "<")
        query = query.replace("=", "==").replace("AND", " and ").replace("OR", " or ")
        def find_data(key, c): return next((x for x in self.financial_data.get(key, []) if x['종목코드']==c), None)
        def repl_series(match):
            metric, yr_str, op, val_str = match.groups()
            yrs = int(yr_str)
            limit = float(val_str)
            real_metric = next((m for m in METRICS_LIST if m.upper() == metric.upper()), None)
            if not real_metric: return "False"
            for i in range(yrs):
                d = find_data(f"D-{i+1}y_data", code)
                if not d or real_metric not in d: return "False"
                try:
                    if float(d[real_metric]) == 0: return "False"
                    if not safe_expr(f"{d[real_metric]} {op} {limit}"): return "False"
                except Exception: return "False"
            return "True"
        def repl_basic(match):
            metric, op, v_str = match.groups()
            real_metric = next((m for m in METRICS_LIST if m.upper() == metric.upper()), None)
            if not real_metric: return match.group(0)
            d = find_data('D-1y_data', code)
            if not d or real_metric not in d: return "False"
            try:
                if float(d[real_metric]) == 0: return "False"
                return f"{d[real_metric]} {op} {v_str}"
            except Exception: return "False"
        try:
            q = re.sub(r"([가-힣a-zA-Z0-9\/]+)\s*\*\s*(\d+)\s*([><=!]+)\s*(-?[\d\.]+)", repl_series, query)
            q = re.sub(r"([가-힣a-zA-Z0-9\/]+)\s*([><=!]+)\s*(-?[\d\.]+)", repl_basic, q)
            return safe_expr(q)
        except Exception: return False

    def run_financial_filter(self, query):
        current_data_key = 'D-1y_data'
        filtered_companies = []
        if current_data_key not in self.financial_data: return filtered_companies
        try:
            for comp in self.financial_data[current_data_key]:
                if self.evaluate_financial_query(comp['종목코드'], query):
                    filtered_companies.append(comp.copy())
        except Exception: pass
        return filtered_companies

    def evaluate_logic(self, logic_str, row):
        if not logic_str or not logic_str.strip(): return False
        expr = logic_str

        for col in row.index:
            if isinstance(col, str) and col.startswith("__CONS_") and col in expr:
                expr = expr.replace(col, str(row[col]))

        sorted_keys = sorted(TECH_INDICATORS_MAP.keys(), key=len, reverse=True)
        for ko in sorted_keys:
            if ko in expr:
                en = TECH_INDICATORS_MAP[ko]
                val = row.get(en, 0)
                if pd.isna(val):
                    val = 0
                expr = expr.replace(ko, str(val))
        expr = expr.replace(">=", "__GE__").replace("<=", "__LE__").replace("==", "__EQ__")
        expr = expr.replace("=", "==")
        expr = expr.replace("__GE__", ">").replace("__LE__", "<").replace("__EQ__", "==")
        expr = expr.replace("AND", "and").replace("OR", "or")
        expr = expr.replace("TRUE", "True").replace("FALSE", "False")
        try:
            return safe_expr(expr)
        except Exception as e:
            raise Exception(f"구문 오류 ({logic_str} -> {expr}): {str(e)}")

    def preprocess_consecutive_logic(self, df, logic_str):
        if not logic_str: return df, logic_str
        pattern = r"([가-힣a-zA-Z0-9_]+)\s*\*\s*(\d+)\s*([><=!]+)\s*(-?[\d\.]+)"
        matches = re.findall(pattern, logic_str)
        for keyword, days_str, op, val_str in matches:
            days = int(days_str)
            val = float(val_str)
            if keyword in TECH_INDICATORS_MAP:
                target_col = TECH_INDICATORS_MAP[keyword]
            else: continue
            if target_col not in df.columns: continue
            temp_col_name = f"__CONS_{keyword}_{days}_{val}__"
            if op == '>': mask = (df[target_col] > val)
            elif op == '>=': mask = (df[target_col] > val)
            elif op == '<': mask = (df[target_col] < val)
            elif op == '<=': mask = (df[target_col] < val)
            elif op == '==': mask = (df[target_col] == val)
            else: continue
            df[temp_col_name] = mask.rolling(window=days).min().fillna(0).astype(bool)
            regex_sub = f"{re.escape(keyword)}\\s*\\*\\s*{days}\\s*{re.escape(op)}\\s*{val_str}"
            logic_str = re.sub(regex_sub, f"{temp_col_name}==True", logic_str)
        return df, logic_str

    def calculate_advanced_stats(self, value_series, trades, wins, profit, loss):
        if value_series.empty: return {}
        start_val = value_series.iloc[0]
        end_val = value_series.iloc[-1]
        total_return = ((end_val - start_val) / start_val) * 100
        try:
            days = (value_series.index[-1] - value_series.index[0]).days
            years = days / 365.25
            if years > 0 and start_val > 0 and end_val > 0:
                cagr = ((end_val / start_val) ** (1/years) - 1) * 100
            else: cagr = 0
        except Exception: cagr = 0
        rolling_max = value_series.cummax()
        drawdown = (value_series - rolling_max) / rolling_max
        mdd = drawdown.min() * 100 
        if trades > 0:
            win_rate_pct = (wins / trades * 100)
            if loss == 0: pf = "Inf"
            else: pf = f"{(profit / loss):.2f}"
            win_rate_str = f"{win_rate_pct:.1f}% (PF: {pf})"
        else:
            win_rate_str = "0.0% (PF: 0)"
        try:
            daily_ret = value_series.pct_change().dropna()
            if daily_ret.std() == 0: sharpe = 0
            else:
                sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        except Exception: sharpe = 0
        return {
            "Total Return": f"{total_return:.2f}%",
            "CAGR": f"{cagr:.2f}%",
            "MDD": f"{mdd:.2f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Win Rate": win_rate_str
        }

    def run_monte_carlo(self, strategy_series, mc_period_str):
        if strategy_series.empty or len(strategy_series) < 2: return {}
        period_map = {"1개월": 21, "3개월": 63, "6개월": 126, "1년": 252, "3년": 756}
        days_to_sim = period_map.get(mc_period_str, 0)
        if days_to_sim == 0: return {}
        log_returns = np.log(strategy_series / strategy_series.shift(1)).dropna()
        if log_returns.empty: return {}
        mu = log_returns.mean()
        sigma = log_returns.std()
        last_price = strategy_series.iloc[-1]
        num_simulations = 1000
        drift = mu - 0.5 * (sigma**2)
        shocks = sigma * np.random.standard_normal((days_to_sim, num_simulations))
        increments = drift + shocks
        cumulative_returns = np.cumsum(increments, axis=0)
        paths = last_price * np.exp(cumulative_returns)
        upper_bound = np.percentile(paths, 95, axis=1)
        median_bound = np.percentile(paths, 50, axis=1)
        lower_bound = np.percentile(paths, 5, axis=1)
        last_date = strategy_series.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_sim + 1)]
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "upper": upper_bound.tolist(),
            "median": median_bound.tolist(),
            "lower": lower_bound.tolist()
        }

    # 휴리스틱 고정 비중 전략 (정교한 최적화 미구현 — 의도된 단순 근사).
    # equal_weight·all_in·inverse_volatility 는 아래에서 실제 계산한다.
    ALLOCATION_CONSTANTS = {
        'risk_parity': 0.10, 'kelly_criterion': 0.15, 'momentum_weight': 0.12,
        'min_variance': 0.08, 'max_sharpe': 0.10, 'market_cap': 0.05,
        'dynamic_asset': 0.20,
    }
    DEFAULT_ALLOCATION = 0.10
    VOL_WINDOW = 60

    def _trailing_vol(self, df, date, window=VOL_WINDOW):
        """date 시점까지 최근 window 거래일의 일간 수익률 표준편차.
        계산 불가(데이터 부족·0·비유한)면 None."""
        try:
            hist = df[df.index <= date]['Close'].tail(window + 1)
            if len(hist) < 3:
                return None
            sd = float(hist.pct_change().dropna().std())
            if not np.isfinite(sd) or sd <= 0:
                return None
            return sd
        except Exception:
            return None

    def _allocation_ratio(self, strategy, code, date, processed_data,
                          holdings, buy_signals_count):
        """매수 1건에 투입할 현재 포트폴리오 대비 비중을 전략별로 결정.

        - equal_weight     : 신호+보유 종목 균등 (기존 동작 보존)
        - all_in           : 전량 집중
        - inverse_volatility: 거래 유니버스 내 변동성 역수 정규화 가중
        - 그 외            : ALLOCATION_CONSTANTS 휴리스틱 고정 비중
        """
        if strategy == 'equal_weight':
            return 1.0 / max(1, buy_signals_count + len(holdings))
        if strategy == 'all_in':
            return 1.0
        if strategy == 'inverse_volatility':
            inv = {}
            for c, info in processed_data.items():
                sd = self._trailing_vol(info['df'], date)
                if sd is not None:
                    inv[c] = 1.0 / sd
            total = sum(inv.values())
            if code not in inv or total <= 0:
                return self.DEFAULT_ALLOCATION
            return inv[code] / total
        return self.ALLOCATION_CONSTANTS.get(strategy, self.DEFAULT_ALLOCATION)

    INITIAL_CAPITAL = 100_000_000
    FIXED_INVESTMENT_PER_STOCK = 10_000_000

    DART_EVENT_KEYWORDS = {
        '유상증자':  'Event_RightIssue',
        '무상증자':  'Event_BonusIssue',
        '자사주취득': 'Event_TreasuryAcq',
        '자사주처분': 'Event_TreasuryDisp',
        '주식분할':  'Event_StockSplit',
        '감자':      'Event_CapitalReduction',
        '합병':      'Event_Merger',
        '타법인주식취득': 'Event_Acquisition',
        '단일판매':  'Event_Supply',
        '공급계약':  'Event_Supply',
        '전환사채':  'Event_CB',
        '신주인수권부사채': 'Event_BW',
        '교환사채':  'Event_EB',
        '배당':      'Event_Dividend',
        '유형자산':  'Event_AssetAcq',
        '영업양수도': 'Event_BizTransfer',
    }

    def _collect_dart_events(self, target_companies, buy_logic, sell_logic,
                             period_months, dart_key, progress_callback=None):
        """식에 DART 공시 키워드가 있을 때만 종목별 이벤트 맵을 수집한다.
        반환: {code: {'YYYY-MM-DD': {'Event_*': 1}, ...}} (없으면 빈 dict)."""
        events_to_fetch = [k for k in self.DART_EVENT_KEYWORDS
                           if k in buy_logic or k in sell_logic]
        event_map = {}
        if not (events_to_fetch and dart_key):
            return event_map
        if progress_callback:
            progress_callback(2, f"DART 공시 데이터 수집 중 ({', '.join(events_to_fetch)})...")
        try:
            # 선택적·무거운 의존성. DART 공시 이벤트가 식에 포함된
            # 경우에만 지연 로드한다(미설치 환경에서도 일반 백테스트 동작).
            import OpenDartReader
            dart = OpenDartReader(dart_key)
            start_dt = (datetime.today() - timedelta(days=period_months * 30 + 100)).strftime("%Y%m%d")
            end_dt   = datetime.today().strftime("%Y%m%d")

            for comp in target_companies:
                code = str(comp['종목코드']).zfill(6)

                # 캐시 확인 — 이미 저장된 이벤트가 있으면 DB에서 로드
                if self.db_manager.has_event_cache(code):
                    cached = self.db_manager.load_event_data(code)
                    if cached:
                        event_map[code] = cached
                    continue

                try:
                    # 전체 공시 조회 (모든 종류의 이벤트 탐색을 위함)
                    reports = dart.list(code, start=start_dt, end=end_dt)
                    code_events = {}
                    if reports is not None and not reports.empty:
                        for _, r in reports.iterrows():
                            report_nm = str(r.get('report_nm', ''))
                            rcept_dt  = str(r.get('rcept_dt', ''))
                            if len(rcept_dt) < 8:
                                continue
                            date_key = f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
                            for keyword, col_name in self.DART_EVENT_KEYWORDS.items():
                                if keyword in events_to_fetch and keyword in report_nm:
                                    code_events.setdefault(date_key, {})[col_name] = 1

                    event_map[code] = code_events
                    # 빈 결과도 캐시에 저장해 중복 API 호출 방지
                    # (빈 dict는 has_event_cache가 False를 반환하므로 sentinel row 삽입)
                    if not code_events:
                        self.db_manager.save_event_data(code, {'__none__': {'__none__': 0}})
                    else:
                        self.db_manager.save_event_data(code, code_events)

                    time.sleep(0.15)  # DART API 속도 제한 대응
                except Exception as e:
                    logger.warning("DART 이벤트 수집 오류 (%s): %s", code, e)
        except Exception as e:
            logger.warning("DART 이벤트 수집 전체 오류: %s", e)
        return event_map

    def _prepare_company_data(self, target_companies, buy_logic, sell_logic,
                              fetch_years, event_map, progress_callback=None):
        """종목별 가격/투자자 데이터를 캐시·크롤링으로 확보하고 지표·권리락
        보정·DART 이벤트를 반영한 분석용 DataFrame 묶음을 만든다."""
        processed_data = {}
        total_comps = len(target_companies)
        for idx, comp in enumerate(target_companies):
            code = str(comp['종목코드']).zfill(6)
            name = comp.get('기업명', code)
            if progress_callback: progress_callback(10 + int((idx/total_comps)*30), f"[{idx+1}/{total_comps}] {name} 데이터 확인 중...")

            latest_price_date_str = self.db_manager.get_latest_date(code, 'price_data')
            latest_price_date = pd.to_datetime(latest_price_date_str) if latest_price_date_str else None

            if latest_price_date is None or latest_price_date.date() < datetime.today().date():
                if progress_callback: progress_callback(None, f"[{idx+1}/{total_comps}] {name} 최신 가격 데이터 크롤링 중 (Naver)...")
                new_price_df = CrawlerUtil.fetch_naver_stock_html(code, years=fetch_years, stop_date=latest_price_date)
                if not new_price_df.empty:
                    new_price_df.set_index('Date', inplace=True)
                    self.db_manager.save_price_data(code, new_price_df)

            price_df = self.db_manager.load_price_data(code)
            if price_df.empty: continue
            # 권리락(액면분할·무상증자·유상증자 등) 보정 — 원주가 갭을 실손익으로 오인하지 않도록
            # 권리락 이전 구간을 소급 조정. (DB의 원주가는 그대로 두고 메모리 시계열만 보정)
            price_df = TechnicalAnalysis.adjust_for_corporate_actions(price_df)

            latest_inv_date_str = self.db_manager.get_latest_date(code, 'investor_data')
            latest_inv_date = pd.to_datetime(latest_inv_date_str) if latest_inv_date_str else None

            if latest_inv_date is None or latest_inv_date.date() < datetime.today().date():
                if progress_callback: progress_callback(None, f"[{idx+1}/{total_comps}] {name} 투자자 데이터 크롤링 중 (Naver)...")
                new_inv_df = CrawlerUtil.fetch_investor_data(code, years=fetch_years, stop_date=latest_inv_date)
                if not new_inv_df.empty:
                    new_inv_df.set_index('Date', inplace=True)
                    self.db_manager.save_investor_data(code, new_inv_df)

            investor_df = self.db_manager.load_investor_data(code)

            if not investor_df.empty:
                df = price_df.join(investor_df, how='left')
                df['Inst_Net_Volume'] = df['Inst_Net_Volume'].fillna(0)
                df['Foreign_Net_Volume'] = df['Foreign_Net_Volume'].fillna(0)
            else:
                df = price_df
                df['Inst_Net_Volume'] = 0
                df['Foreign_Net_Volume'] = 0

            df = df.sort_index()
            df = TechnicalAnalysis.add_indicators(df)

            # DART 이벤트를 해당 날짜 행에 반영
            if code in event_map:
                for date_str, cols in event_map[code].items():
                    try:
                        ts = pd.Timestamp(date_str)
                        if ts in df.index:
                            for col, val in cols.items():
                                if col in df.columns:
                                    df.loc[ts, col] = val
                    except Exception:
                        pass

            df, b_logic = self.preprocess_consecutive_logic(df, buy_logic)
            df, s_logic = self.preprocess_consecutive_logic(df, sell_logic)

            processed_data[code] = {"df": df, "b_logic": b_logic, "s_logic": s_logic}
        return processed_data

    def _run_daily_simulation(self, processed_data, sim_dates, portfolio_strategy,
                              use_tax_fee, name_by_code, progress_callback=None):
        """일별 매수/매도 시뮬레이션. 전략 곡선·벤치마크 곡선·체결 내역과
        승률 통계를 담은 dict 를 반환한다(로직 오류 시 {'error': ...})."""
        portfolio_curve = []
        benchmark_curve = []

        cash = self.INITIAL_CAPITAL
        holdings = {}
        buy_prices = {}
        trade_count = 0
        win_count = 0
        total_profit = 0
        total_loss = 0

        bnh_holdings = {}
        bnh_cost_basis = 0
        trade_history_logs = []

        COMMISSION_RATE = 0.00015 if use_tax_fee else 0.0
        TRANSACTION_TAX = 0.0018 if use_tax_fee else 0.0

        for d_idx, date in enumerate(sim_dates):
            if d_idx % (len(sim_dates)//10 + 1) == 0 and progress_callback:
                pct = 45 + int((d_idx/len(sim_dates)) * 40)
                progress_callback(pct, f"[{date.strftime('%Y-%m-%d')}] 수익률 평가 및 매매 판단 중...")

            for code in processed_data.keys():
                if code not in bnh_holdings:
                    df = processed_data[code]['df']
                    if date in df.index:
                        price = df.loc[date, 'Close']
                        if price > 0:
                            qty = int(self.FIXED_INVESTMENT_PER_STOCK // price)
                            cost = int(qty * price)
                            if qty > 0:
                                bnh_holdings[code] = qty
                                bnh_cost_basis += cost
            bnh_market_value = 0
            for code, qty in bnh_holdings.items():
                current_price = price_on_or_before(processed_data[code]['df'], date)
                bnh_market_value += qty * current_price

            if bnh_cost_basis > 0:
                bnh_return_rate = (bnh_market_value - bnh_cost_basis) / bnh_cost_basis
                display_bnh_value = int(self.INITIAL_CAPITAL * (1 + bnh_return_rate))
            else: display_bnh_value = self.INITIAL_CAPITAL
            benchmark_curve.append({'Date': date, 'BnH': display_bnh_value})

            current_pf_value = cash
            for code, qty in holdings.items():
                price = price_on_or_before(processed_data[code]['df'], date)
                current_pf_value += qty * price

            portfolio_curve.append({'Date': date, 'Strategy': current_pf_value})

            # Process Sells
            for code in list(holdings.keys()):
                if date not in processed_data[code]['df'].index: continue
                row = processed_data[code]['df'].loc[date]
                s_logic = processed_data[code]['s_logic']

                try:
                    if self.evaluate_logic(s_logic, row):
                        qty = holdings[code]
                        price = row['Close']
                        buy_price = buy_prices.get(code, price)
                        sell_amt_gross = int(qty * price)
                        sell_fee = int(sell_amt_gross * (COMMISSION_RATE + TRANSACTION_TAX))
                        sell_amt_net = sell_amt_gross - sell_fee

                        buy_amt_net = int(qty * buy_price) + int(qty * buy_price * COMMISSION_RATE)
                        profit = sell_amt_net - buy_amt_net

                        cash += sell_amt_net
                        trade_count += 1

                        comp_name = name_by_code.get(code, code)
                        trade_history_logs.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'type': '매도',
                            'code': code,
                            'name': comp_name,
                            'price': int(price),
                            'qty': qty,
                            'total': int(sell_amt_net),
                            'profit': int(profit),
                            'return': f"{(profit/max(1, buy_amt_net)*100):.2f}%"
                        })

                        if profit > 0:
                            win_count += 1
                            total_profit += profit
                        else:
                            total_loss += abs(profit)
                        del holdings[code]
                        if code in buy_prices: del buy_prices[code]
                except Exception as e:
                    return {"error": f"매도 처리 중 에러 발생: {str(e)}"}

            # Process Buys
            buy_signals = []
            for code in processed_data.keys():
                if code in holdings: continue
                if date not in processed_data[code]['df'].index: continue

                row = processed_data[code]['df'].loc[date]
                b_logic = processed_data[code]['b_logic']

                try:
                    if self.evaluate_logic(b_logic, row):
                        buy_signals.append((code, row))
                except Exception as e:
                    return {"error": f"매수 처리 중 에러 발생: {str(e)}"}

            if buy_signals:
                available_slots = 10 - len(holdings) if portfolio_strategy != 'all_in' else 1
                if available_slots > 0:
                    signals_to_execute = buy_signals[:available_slots]

                    for code, row in signals_to_execute:
                        current_pf = cash
                        for hc, hq in holdings.items():
                            p = price_on_or_before(processed_data[hc]['df'], date)
                            current_pf += hq * p

                        # 포트폴리오 배분 전략 할당 (전략별 디스패치)
                        alloc_ratio = self._allocation_ratio(
                            portfolio_strategy, code, date,
                            processed_data=processed_data,
                            holdings=holdings,
                            buy_signals_count=len(buy_signals),
                        )

                        invest_amount_gross = int(current_pf * alloc_ratio)
                        if invest_amount_gross > cash: invest_amount_gross = cash

                        price = row['Close']
                        if pd.isna(price) or price <= 0:
                            continue  # 가격 데이터 없음 — 매수 스킵
                        cost_per_share_incl_fee = price * (1 + COMMISSION_RATE)
                        qty = int(invest_amount_gross // cost_per_share_incl_fee)

                        if qty > 0:
                            holdings[code] = qty
                            total_cost = int(qty * cost_per_share_incl_fee)
                            cash -= total_cost
                            buy_prices[code] = price
                            comp_name = name_by_code.get(code, code)
                            trade_history_logs.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'type': '매수',
                                'code': code,
                                'name': comp_name,
                                'price': int(price),
                                'qty': qty,
                                'total': total_cost
                            })

        return {
            "portfolio_curve": portfolio_curve,
            "benchmark_curve": benchmark_curve,
            "trade_count": trade_count,
            "win_count": win_count,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "trade_history_logs": trade_history_logs,
        }

    def _detect_recent_signals(self, processed_data, name_by_code):
        """각 종목 최근 5거래일에서 매수/매도 신호를 스캔한다."""
        recent_signals = {'buy': [], 'sell': []}
        for code, info in processed_data.items():
            df = info['df']
            if df.empty or len(df) < 5: continue
            comp_name = name_by_code.get(code, code)
            recent_df = df.iloc[-5:]
            for dt, row in recent_df.iterrows():
                try:
                    if self.evaluate_logic(info['b_logic'], row):
                        recent_signals['buy'].append({'date': dt.strftime('%Y-%m-%d'), 'code': code, 'name': comp_name, 'price': int(row['Close'])})
                    if self.evaluate_logic(info['s_logic'], row):
                        recent_signals['sell'].append({'date': dt.strftime('%Y-%m-%d'), 'code': code, 'name': comp_name, 'price': int(row['Close'])})
                except Exception as e:
                    return {"error": f"최근 신호 감지 중 로직 에러 발생: {str(e)}"}
        return recent_signals

    def run_backtest_logic(self, target_companies, buy_logic, sell_logic, period_months, mc_period_str, portfolio_strategy='equal_weight', use_tax_fee=False, dart_key='', progress_callback=None):
        fetch_years = (period_months / 12) + 2
        name_by_code = {str(c['종목코드']).zfill(6): c.get('기업명', str(c['종목코드']).zfill(6))
                        for c in target_companies}

        event_map = self._collect_dart_events(
            target_companies, buy_logic, sell_logic,
            period_months, dart_key, progress_callback)

        if progress_callback: progress_callback(5, f"KOSPI 지수 및 기초 데이터 수집 중...")
        kospi_df = CrawlerUtil.fetch_kospi_index(years=fetch_years)
        if not kospi_df.empty:
            kospi_df = kospi_df[['Close']].rename(columns={'Close': 'KOSPI'})

        processed_data = self._prepare_company_data(
            target_companies, buy_logic, sell_logic,
            fetch_years, event_map, progress_callback)

        if not processed_data:
            return {"error": "데이터 수집 실패"}

        all_dates_raw = sorted(list(set().union(*[v['df'].index for v in processed_data.values()])))
        end_date = datetime.today()
        start_date = end_date - timedelta(days=period_months * 30)
        sim_dates = [d for d in all_dates_raw if start_date <= d <= end_date]

        if not sim_dates:
            return {"error": "선택한 기간에 해당하는 데이터가 충분하지 않습니다."}

        if progress_callback: progress_callback(45, "데이터 수집 완료. 백테스트 일별 시뮬레이션을 시작합니다...")

        sim = self._run_daily_simulation(
            processed_data, sim_dates, portfolio_strategy,
            use_tax_fee, name_by_code, progress_callback)
        if "error" in sim:
            return sim

        result_df = pd.DataFrame(sim['portfolio_curve']).set_index('Date')
        bnh_df = pd.DataFrame(sim['benchmark_curve']).set_index('Date')
        result_df = result_df[~result_df.index.duplicated(keep='last')]
        bnh_df = bnh_df[~bnh_df.index.duplicated(keep='last')]

        final_df = result_df.rename(columns={'Strategy': 'Strategy_Value'}).join(bnh_df.rename(columns={'BnH': 'BnH_Value'}), how='outer')
        final_df = final_df.ffill().fillna(self.INITIAL_CAPITAL)

        final_df['Strategy'] = ((final_df['Strategy_Value'] - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL) * 100
        final_df['BnH'] = ((final_df['BnH_Value'] - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL) * 100

        if not kospi_df.empty:
            kospi_cut = kospi_df[kospi_df.index >= sim_dates[0]].copy()
            if not kospi_cut.empty:
                base_k = kospi_cut.iloc[0]['KOSPI']
                kospi_cut['KOSPI_Pct'] = ((kospi_cut['KOSPI'] - base_k) / base_k) * 100
                final_df = final_df.join(kospi_cut[['KOSPI_Pct']], how='left')
                final_df = final_df.rename(columns={'KOSPI_Pct': 'KOSPI'})

        final_df = final_df.ffill().fillna(0)

        final_bnh_pct = final_df.iloc[-1]['BnH'] if not final_df.empty else 0
        final_kospi_pct = final_df.iloc[-1]['KOSPI'] if not final_df.empty and 'KOSPI' in final_df.columns else 0

        detailed_stats = self.calculate_advanced_stats(final_df['Strategy_Value'], sim['trade_count'], sim['win_count'], sim['total_profit'], sim['total_loss'])

        detailed_stats["BnH Return"] = f"{final_bnh_pct:.2f}%"
        detailed_stats["KOSPI Return"] = f"{final_kospi_pct:.2f}%"

        mc_results = {}
        if mc_period_str and mc_period_str != "안함":
            mc_results = self.run_monte_carlo(final_df['Strategy_Value'], mc_period_str)

        recent_signals = self._detect_recent_signals(processed_data, name_by_code)
        if "error" in recent_signals:
            return recent_signals

        # Prepare for JSON
        final_df.reset_index(inplace=True)
        final_df['Date'] = final_df['Date'].astype(str)
        final_data = final_df.to_dict(orient='records')

        return {
            "detailed_stats": detailed_stats,
            "returns_data": final_data,
            "mc_results": mc_results,
            "recent_signals": recent_signals,
            "trade_history": sim['trade_history_logs'],
        }
