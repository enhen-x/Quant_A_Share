import baostock as bs
import pandas as pd
import os
import datetime
import time
import socket
from typing import List, Tuple
from tqdm import tqdm

# 全局网络超时（秒）
socket.setdefaulttimeout(20)

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')


# --- 通用重试工具 ---
def _sleep_backoff(attempt: int, base: float = 1.0, factor: float = 1.6, jitter: float = 0.3) -> None:
    delay = base * (factor ** attempt)
    delay *= (1.0 + (jitter * (2 * (time.perf_counter() % 1) - 1)))  # 简单抖动
    time.sleep(min(delay, 8.0))


def _need_relogin(exc: Exception) -> bool:
    msg = str(exc)
    # 常见网络断开/重置 + baostock 常见提示
    return any(k in msg for k in [
        'WinError 10054', '10054', 'WinError 10053',
        'Connection reset', 'Connection aborted', 'Broken pipe',
        'timed out', 'read timed out', 'EOFError',
        '远程主机强迫关闭', '接收数据异常', '请稍后再试'
    ])


def _with_retry(call, max_retries: int = 5, op_name: str = ""):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call()
        except Exception as e:
            last_exc = e
            if _need_relogin(e):
                try:
                    bs.logout()
                except Exception:
                    pass
                _sleep_backoff(attempt)
                lg = bs.login()
                if getattr(lg, "error_code", "0") != '0':
                    _sleep_backoff(attempt)
                continue
            else:
                _sleep_backoff(attempt)
                continue
    raise last_exc if last_exc else RuntimeError(f"{op_name} failed")


def _nearest_trading_day(max_back_days: int = 10) -> str:
    """向后回溯找到最近可用的交易日字符串。"""
    check_date = datetime.datetime.now()
    for i in range(max_back_days):
        date_str = check_date.strftime("%Y-%m-%d")

        def _query():
            return bs.query_all_stock(day=date_str)

        try:
            rs = _with_retry(_query, op_name="query_all_stock")
        except Exception:
            check_date -= datetime.timedelta(days=1)
            continue

        has_any = (rs.error_code == '0')
        ok = False
        try:
            while has_any and rs.next():
                if rs.get_row_data():
                    ok = True
                    break
        except Exception:
            # 本次失败则回退一天重试
            ok = False

        if ok:
            return date_str
        check_date -= datetime.timedelta(days=1)
    # 兜底：今天
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _daterange_chunks(start_date: str, end_date: str, chunk_days: int = 90) -> List[Tuple[str, str]]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        return []
    spans = []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        spans.append((cur.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        cur = nxt + pd.Timedelta(days=1)
    return spans


def _fetch_history(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    稳健拉取：将区间按90天切片；对每个切片，若在 rs.next() 中途断线，
    自动以“最后成功日期+1天”为新起点继续补齐该切片剩余数据。
    """
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        return pd.DataFrame(columns=[
            'date','code','open','high','low','close','volume','amount','turn','pctChg'
        ])

    segments = _daterange_chunks(start_date, end_date, chunk_days=30)
    frames = []

    for seg_start, seg_end in segments:
        cur_start = seg_start
        # 针对该切片的“断点续传”循环
        seg_guard_attempts = 0
        seg_rows = []

        while pd.to_datetime(cur_start) <= pd.to_datetime(seg_end):
            def _query():
                return bs.query_history_k_data_plus(
                    code,
                    "date,code,open,high,low,close,volume,amount,turn,pctChg",
                    start_date=cur_start,
                    end_date=seg_end,
                    frequency="d",
                    adjustflag="2",
                )

            try:
                rs = _with_retry(_query, op_name="query_history_k_data_plus")
            except Exception as e:
                # 本段起点请求失败，跳过到下一小段（避免死循环）
                seg_guard_attempts += 1
                if seg_guard_attempts >= 6:
                    break
                _sleep_backoff(seg_guard_attempts)
                continue

            last_ok_date = None
            try:
                # 若返回即错误，也触发重试
                if getattr(rs, "error_code", "0") != '0':
                    raise RuntimeError(getattr(rs, "error_msg", "query error"))

                while rs.next():
                    row = rs.get_row_data()
                    if not row:
                        continue
                    seg_rows.append(row)
                    # Baostock fields 次序与 rs.fields 对齐，date 在索引 0
                    last_ok_date = row[0]
            except Exception as e:
                # 迭代中途失败：从 last_ok_date+1 继续补
                if last_ok_date:
                    cur_start = (pd.to_datetime(last_ok_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    # 没拿到任何一行，说明起点就失败；指数退避后重试
                    seg_guard_attempts += 1
                    if seg_guard_attempts >= 6:
                        break
                    _sleep_backoff(seg_guard_attempts)
                    # 重新开始本次 while（cur_start 不变）
                    continue
            else:
                # 本次子区间成功跑完，退出该切片的 while
                break

        if seg_rows:
            df = pd.DataFrame(seg_rows, columns=rs.fields)
            frames.append(df)

        # 轻微节流，避免过快触发服务端限流
        time.sleep(0.25)

    if not frames:
        return pd.DataFrame(columns=[
            'date','code','open','high','low','close','volume','amount','turn','pctChg'
        ])

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if df.empty:
        return df

    # 类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).drop_duplicates(subset=['date'], keep='last').sort_values('date')
    return df


def _atomic_write_csv(df: pd.DataFrame, file_path: str) -> None:
    tmp_path = file_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, file_path)  # 原子替换，避免中途失败损坏文件


def _update_or_download_single(code: str, start_date_default: str, end_date_str: str) -> bool:
    file_path = os.path.join(RAW_DATA_DIR, f"{code}.csv")
    end_date_dt = pd.to_datetime(end_date_str)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
        try:
            existing = pd.read_csv(file_path)
            if 'date' not in existing.columns:
                return False
            existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
            last_date = existing['date'].max()

            # 如果文件没有有效日期，回退全量
            if pd.isna(last_date):
                start_date = start_date_default
            else:
                # 从下一天开始补齐
                start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            # 如果最后日期已经覆盖到结束日期，则不更新
            if not pd.isna(last_date) and last_date >= end_date_dt:
                return True

            # 如果起始超过结束，直接返回
            if pd.to_datetime(start_date) > end_date_dt:
                return True

            new_df = _fetch_history(code, start_date, end_date_str)
            if new_df.empty:
                # 可能停牌或无新数据
                return True

            merged = pd.concat([existing, new_df], ignore_index=True)
            if 'date' in merged.columns:
                merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
                merged = merged.dropna(subset=['date']).drop_duplicates(subset=['date'], keep='last').sort_values('date')
            _atomic_write_csv(merged, file_path)
            return True
        except Exception:
            return False
    else:
        try:
            full_df = _fetch_history(code, start_date_default, end_date_str)
            if full_df.empty:
                return False
            if 'date' in full_df.columns:
                full_df.sort_values('date', inplace=True)
            _atomic_write_csv(full_df, file_path)
            return True
        except Exception:
            return False


def _list_local_codes() -> list:
    """从本地 raw 目录扫描已有 CSV 推断股票代码列表。"""
    if not os.path.isdir(RAW_DATA_DIR):
        return []
    codes = []
    for fn in os.listdir(RAW_DATA_DIR):
        if fn.endswith(".csv"):
            codes.append(os.path.splitext(fn)[0])
    # 过滤合法的 baostock 代码格式
    codes = [c for c in codes if c.startswith(('sh.', 'sz.', 'bj.'))]
    return sorted(set(codes))


def _list_market_codes(end_date_str: str) -> list:
    """从 baostock 拉取市场代码（一次），带重试。"""
    def _query():
        return bs.query_all_stock(day=end_date_str)

    rows = []
    try:
        rs = _with_retry(_query, op_name="query_all_stock(list)")
    except Exception:
        return []

    try:
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
    except Exception:
        # 单次失败则返回已拿到部分或空
        pass

    if not rows:
        return []
    df_stocks = pd.DataFrame(rows, columns=rs.fields)
    df_stocks['code'] = df_stocks['code'].astype(str)
    target = df_stocks[df_stocks['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]['code'].tolist()
    # 排除科创板（688）和部分板块示例，可按需调整
    target = [code for code in target if not code.startswith(('sh.688', 'bj', 'sz.8', 'sz.4'))]
    return target


def download_all_stock_history(
    start_date: str = "2014-01-01",
    codes: list | None = None,
    prefer_local: bool = True,
    include_new: bool = True
):
    """
    稳健增量下载/更新：
    - prefer_local=True：优先根据本地已有 CSV 增量更新；
    - include_new=True：在本地基础上补充市场新股；
    - codes 指定则仅更新该列表。
    """
    # 目录
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"创建数据目录: {RAW_DATA_DIR}")

    lg = bs.login()
    if lg.error_code != '0':
        print(f"登陆失败: {lg.error_msg}")
        return

    # 使用最近交易日作为结束日期
    end_date_str = _nearest_trading_day()

    # 决定待处理代码列表
    final_codes = []
    if codes:
        final_codes = list(sorted(set(codes)))
        print(f"使用用户指定代码列表，共 {len(final_codes)} 只。")
    elif prefer_local:
        local_codes = _list_local_codes()
        final_codes = list(local_codes)
        if include_new:
            market_codes = _list_market_codes(end_date_str)
            if market_codes:
                # 合并本地与新股
                s = set(final_codes)
                add = [c for c in market_codes if c not in s]
                final_codes.extend(add)
        if final_codes:
            print(f"基于本地增量更新（含新股={include_new}），共 {len(final_codes)} 只。")
        else:
            market_codes = _list_market_codes(end_date_str)
            final_codes = market_codes
            print(f"本地无数据，回退市场列表，共 {len(final_codes)} 只。")
    else:
        market_codes = _list_market_codes(end_date_str)
        final_codes = market_codes
        print(f"使用市场列表，共 {len(final_codes)} 只。")

    if not final_codes:
        print("未获取到待更新的股票代码。")
        bs.logout()
        return

    print(f"结束日期: {end_date_str}，开始更新/下载...")
    updated, failed = 0, 0
    for code in tqdm(final_codes, desc="更新进度"):
        ok = _update_or_download_single(code, start_date, end_date_str)
        if ok:
            updated += 1
        else:
            failed += 1

    bs.logout()
    print("\n" + "="*30)
    print("任务完成！")
    print(f"成功写入(包含全量/增量): {updated}")
    print(f"失败或未写入: {failed}")
    print(f"存储位置: {RAW_DATA_DIR}")
    print("="*30)


if __name__ == "__main__":
    # 示例：仅增量更新本地已有，并自动补充新股
    download_all_stock_history(start_date="2014-01-01", prefer_local=True, include_new=True)