"""Load annual reports into the knowledge base.

  python ingest.py                     index every PDF in data/reports/ named COMPANY_YEAR.pdf (e.g. TCS_2024.pdf)
  python ingest.py --nse TCS           download the latest annual report of an Indian company from NSE and index it
  python ingest.py --nse TCS --year 2025   the report for the financial year ending in 2025
  python ingest.py --edgar AAPL        download the latest 10-K for a US company from SEC EDGAR and index it
"""
import argparse
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

import rag

REPORTS_DIR = Path(__file__).parent / "data" / "reports"

# NSE has no official API. This is the JSON endpoint its own website calls; it only answers a
# browser-like session that has loaded a page first and holds the cookies from it.
NSE_PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
NSE_API = "https://www.nseindia.com/api/annual-reports"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_PAGE,
}
NSE_BLOCKED = "NSE did not answer (it sometimes blocks automated requests). Try again, or upload the PDF yourself."


def _progress(done, total):
    print(f"\r  embedded {done}/{total} chunks", end="", flush=True)


def ingest_pdfs():
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {REPORTS_DIR}. Add files named COMPANY_YEAR.pdf, e.g. TCS_2024.pdf")
        return
    for pdf in pdfs:
        match = re.fullmatch(r"(.+)_(\d{4})", pdf.stem)
        if not match:
            print(f"Skipping {pdf.name}: name must look like COMPANY_YEAR.pdf")
            continue
        company, year = match.group(1), int(match.group(2))
        print(f"{pdf.name} -> company={company.upper()} year={year}")
        n = rag.add_report(company, year, rag.pdf_pages(pdf), source=pdf.name, progress=_progress)
        print(f"\n  stored {n} chunks")


def _nse_session():
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    session.get(NSE_PAGE, timeout=30)
    return session


def nse_symbol(symbol):
    """'tcs.ns' -> 'TCS': accept the Yahoo Finance form of an NSE symbol too."""
    return re.sub(r"\.(NS|BO)$", "", symbol.strip().upper())


def nse_reports(symbol, session=None):
    """Annual reports NSE lists for a symbol, newest first.

    Each is {"year", "label", "url"}; year is the one the financial year ends in (FY 2024-25 -> 2025).
    """
    symbol = nse_symbol(symbol)
    session = session or _nse_session()
    try:
        response = session.get(NSE_API, params={"index": "equities", "symbol": symbol}, timeout=30)
        response.raise_for_status()
        rows = response.json().get("data") or []
    except (requests.RequestException, ValueError) as e:
        raise ValueError(NSE_BLOCKED) from e
    reports = [
        {"year": int(r["toYr"]), "label": f"FY {r['fromYr']}-{r['toYr'][-2:]}", "url": r["fileName"]}
        for r in rows
        if r.get("fileName") and str(r.get("toYr", "")).isdigit()
    ]
    if not reports:
        raise ValueError(f"NSE lists no annual reports for '{symbol}'. Use the NSE symbol, e.g. TCS, INFY, RELIANCE.")
    # NSE lists the newest filing first, so a revised report wins over the one it replaces.
    latest = {}
    for report in reports:
        latest.setdefault(report["year"], report)
    return sorted(latest.values(), key=lambda r: r["year"], reverse=True)


def _pdf_bytes(data):
    """NSE serves older reports as a zip holding the PDF."""
    if data[:4] != b"PK\x03\x04":
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        pdfs = [f for f in archive.infolist() if f.filename.lower().endswith(".pdf")]
        if not pdfs:
            raise ValueError("The NSE download has no PDF inside.")
        return archive.read(max(pdfs, key=lambda f: f.file_size))


def ingest_nse(symbol, year=None, progress=None):
    """Download one annual report from NSE (the latest if no year) and index it.

    Returns (company, year, chunks stored).
    """
    symbol = nse_symbol(symbol)
    session = _nse_session()
    reports = nse_reports(symbol, session)
    if year:
        report = next((r for r in reports if r["year"] == int(year)), None)
        if not report:
            years = ", ".join(str(r["year"]) for r in reports)
            raise ValueError(f"NSE has no {symbol} report for {year}. Available years: {years}.")
    else:
        report = reports[0]
    try:
        response = session.get(report["url"], timeout=180)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(NSE_BLOCKED) from e
    n = rag.add_report(symbol, report["year"], rag.pdf_pages(_pdf_bytes(response.content)),
                       source=report["url"].rsplit("/", 1)[-1], progress=progress)
    return symbol, report["year"], n


def ingest_edgar(ticker, identity=None, progress=None):
    """Download the latest 10-K from SEC EDGAR and index it. Returns (company, year, chunks stored)."""
    from edgar import Company, set_identity

    identity = identity or os.getenv("EDGAR_IDENTITY")
    if not identity:
        raise ValueError("Set EDGAR_IDENTITY='Your Name your@email.com' in .env (SEC requires it).")
    set_identity(identity)

    ticker = ticker.strip().upper()
    filings = Company(ticker).get_filings(form="10-K")
    if not filings:
        raise ValueError(f"SEC EDGAR has no 10-K for '{ticker}'. It covers US-listed companies only.")
    filing = filings.latest()
    year = int(str(filing.period_of_report or filing.filing_date)[:4])
    # A 10-K is one long text with no pages, so every chunk gets page 0.
    n = rag.add_report(ticker, year, [(0, filing.text())], source=f"SEC EDGAR {filing.accession_no}",
                       progress=progress)
    return ticker, year, n


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nse", metavar="SYMBOL", help="fetch an annual report for an NSE symbol")
    parser.add_argument("--year", type=int, help="with --nse: the year the financial year ends in (default: latest)")
    parser.add_argument("--edgar", metavar="TICKER", help="fetch the latest 10-K for a US ticker from SEC EDGAR")
    args = parser.parse_args()

    try:
        if args.nse:
            company, year, n = ingest_nse(args.nse, args.year, progress=_progress)
            print(f"\n{company} {year}: stored {n} chunks")
        elif args.edgar:
            company, year, n = ingest_edgar(args.edgar, progress=_progress)
            print(f"\n{company} {year}: stored {n} chunks")
        else:
            ingest_pdfs()
    except ValueError as e:
        sys.exit(str(e))
    print("Indexed reports:", ", ".join(f"{c} {y}" for c, y in rag.list_reports()) or "none")
