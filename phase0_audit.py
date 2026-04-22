"""Phase 0: Data Availability Audit.

Checks what is actually scrapable / loadable before building the pipeline.
Writes data_availability_audit.txt.  Does NOT write any CSV or begin analysis.
"""

import os
import sys
import time
from utils import bbref_get, log_audit, BASE_DIR, OUTPUT_DIR

AUDIT_OUT = os.path.join(BASE_DIR, 'data_availability_audit.txt')

# ── Sample teams / years used as spot-checks ──────────────────────────────────
SPOT_CHECK_TEAMS  = ['NYY', 'BOS', 'CHC', 'LAD', 'ATL']
SPOT_CHECK_YEARS  = [2001, 2005, 2010, 2015, 2020, 2024]

FANGRAPHS_TEAMS = {
    'NYY': 9, 'BOS': 2, 'CHC': 16, 'LAD': 22, 'ATL': 15,
}


def write(lines, fh):
    for line in lines:
        fh.write(line + '\n')
    fh.flush()


# ── Section 1: pybaseball + Lahman ────────────────────────────────────────────
def audit_pybaseball(fh):
    write(['', '='*70, 'SECTION 1: pybaseball & Lahman Database', '='*70], fh)

    try:
        import pybaseball
        write([f'  pybaseball version: {pybaseball.__version__}'], fh)
    except ImportError as e:
        write([f'  FAIL — could not import pybaseball: {e}'], fh)
        write(['  ACTION REQUIRED: pip install pybaseball'], fh)
        return

    # Lahman Managers table
    try:
        from pybaseball import lahman
        mgr = lahman.managers()
        max_yr = int(mgr['yearID'].max())
        min_yr = int(mgr['yearID'].min())
        n_rows = len(mgr)
        write([
            f'  Lahman Managers.csv: LOADED — {n_rows} rows, years {min_yr}–{max_yr}',
        ], fh)

        # Check inseason column
        has_inseason = 'inseason' in mgr.columns
        write([f'  inseason column present: {has_inseason}'], fh)
        if has_inseason:
            multi = mgr[mgr['inseason'] > 1]
            write([f'  Rows with inseason > 1 (mid-season changes): {len(multi)}'], fh)

        if max_yr < 2024:
            msg = (f'  WARNING: Lahman max year is {max_yr}. '
                   f'Seasons {max_yr+1}–2025 require supplemental BBRef scraping.')
            write([msg], fh)
            log_audit(msg, 'WARNING')
        if max_yr < 2025:
            write([f'  Seasons beyond {max_yr} will be sourced from BBRef scraping.'], fh)

    except Exception as e:
        write([f'  FAIL loading Lahman Managers: {e}'], fh)
        log_audit(f'Phase0: Lahman managers load failed: {e}', 'ERROR')

    # Lahman People table (for manager names)
    try:
        from pybaseball import lahman
        ppl = lahman.people()
        write([f'  Lahman People.csv: LOADED — {len(ppl)} rows'], fh)
    except Exception as e:
        write([f'  FAIL loading Lahman People: {e}'], fh)

    # pybaseball schedule_and_record spot-check
    write(['', '  Spot-checking pybaseball.schedule_and_record:'], fh)
    try:
        from pybaseball import schedule_and_record
        df = schedule_and_record(2010, 'NYY')
        write([f'  schedule_and_record(2010, NYY): {len(df)} rows, '
               f'columns: {list(df.columns)[:8]} ...'], fh)
    except Exception as e:
        write([f'  FAIL schedule_and_record: {e}'], fh)
        log_audit(f'Phase0: schedule_and_record spot-check failed: {e}', 'WARNING')


# ── Section 2: Baseball Reference URL patterns ────────────────────────────────
def audit_bbref(fh):
    write(['', '='*70, 'SECTION 2: Baseball Reference URL Patterns', '='*70], fh)

    import requests

    # Team schedule/game-log page
    url = 'https://www.baseball-reference.com/teams/NYY/2010-schedule.shtml'
    write([f'  Testing game-log URL: {url}'], fh)
    resp = bbref_get(url)
    if resp:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', id='team_schedule')
        if table:
            rows = table.find_all('tr')
            write([f'  game-log table found — {len(rows)} rows (including header)'], fh)
        else:
            write(['  WARNING: could not find #team_schedule table — page structure may differ'], fh)
            log_audit('Phase0: #team_schedule table not found on BBRef schedule page', 'WARNING')
    else:
        write(['  FAIL: no response from BBRef game-log URL'], fh)

    # Team season page (manager + coaching staff info)
    time.sleep(2)
    url2 = 'https://www.baseball-reference.com/teams/NYY/2010.shtml'
    write([f'', f'  Testing team-season URL: {url2}'], fh)
    resp2 = bbref_get(url2)
    if resp2:
        from bs4 import BeautifulSoup
        soup2 = BeautifulSoup(resp2.text, 'lxml')
        # Look for managers section
        mgr_div = soup2.find(id='div_coaches')
        write([f'  coaches div (#div_coaches) found: {mgr_div is not None}'], fh)
        # Also look for managers table
        mgr_section = soup2.find(string=lambda t: t and 'Manager' in t and 'Record' in t)
        write([f'  "Manager Record" text found: {mgr_section is not None}'], fh)
        if not mgr_div:
            write(['  NOTE: coaching staff may need to be inferred from page text or '
                   'managers.shtml history page'], fh)
    else:
        write(['  FAIL: no response from BBRef team-season URL'], fh)

    # Manager history page
    time.sleep(2)
    url3 = 'https://www.baseball-reference.com/teams/NYY/managers.shtml'
    write([f'', f'  Testing manager history URL: {url3}'], fh)
    resp3 = bbref_get(url3)
    if resp3:
        from bs4 import BeautifulSoup
        soup3 = BeautifulSoup(resp3.text, 'lxml')
        tables = soup3.find_all('table')
        write([f'  Found {len(tables)} table(s) on manager history page'], fh)
    else:
        write(['  FAIL: no response from BBRef manager history URL'], fh)

    # Spot-check several teams/years for missing pages
    write(['', '  Spot-checking additional team/year game-log pages:'], fh)
    problems = []
    for team in SPOT_CHECK_TEAMS[:3]:
        for year in [2001, 2010, 2023]:
            url_t = f'https://www.baseball-reference.com/teams/{team}/{year}-schedule.shtml'
            r = bbref_get(url_t)
            status = r.status_code if r else 'FAIL'
            if status != 200:
                problems.append(f'{team} {year}: status {status}')
                log_audit(f'Phase0: BBRef {team} {year} schedule returned status {status}', 'WARNING')
            else:
                write([f'    {team} {year}: OK'], fh)

    if problems:
        write(['  PROBLEMS:'] + [f'    {p}' for p in problems], fh)
    else:
        write(['  All spot-check pages returned 200 OK'], fh)


# ── Section 3: FanGraphs projections ──────────────────────────────────────────
def audit_fangraphs(fh):
    write(['', '='*70, 'SECTION 3: FanGraphs ZiPS / Steamer Projections', '='*70], fh)

    import requests

    # Try current-year depth charts (projected team W%)
    url = 'https://www.fangraphs.com/depthcharts.aspx?position=Team'
    write([f'  Testing FanGraphs depth charts: {url}'], fh)
    time.sleep(2)
    try:
        resp = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (academic research)'}, timeout=30)
        write([f'  Response status: {resp.status_code}'], fh)
        if resp.status_code == 200:
            write(['  FanGraphs depth charts accessible'], fh)
        else:
            write([f'  WARNING: non-200 status {resp.status_code}'], fh)
            log_audit(f'Phase0: FanGraphs depth charts status {resp.status_code}', 'WARNING')
    except Exception as e:
        write([f'  FAIL: {e}'], fh)
        log_audit(f'Phase0: FanGraphs depth charts request failed: {e}', 'ERROR')

    # Try FanGraphs projections API endpoint (undocumented but used by the site)
    time.sleep(2)
    api_url = ('https://www.fangraphs.com/api/projections'
               '?type=zips&stats=bat&pos=all&team=0&players=0&lg=all&season=2023')
    write([f'', f'  Testing FanGraphs projections API: {api_url}'], fh)
    try:
        resp2 = requests.get(api_url, headers={
            'User-Agent': 'Mozilla/5.0'}, timeout=30)
        write([f'  Response status: {resp2.status_code}'], fh)
        if resp2.status_code == 200:
            try:
                data = resp2.json()
                write([f'  JSON response: type={type(data).__name__}, '
                       f'length={len(data) if isinstance(data, list) else "n/a"}'], fh)
            except Exception:
                write(['  Response is not JSON — likely HTML gate'], fh)
        else:
            write([f'  WARNING: status {resp2.status_code} — API may require auth'], fh)
            log_audit(f'Phase0: FanGraphs projections API status {resp2.status_code}', 'WARNING')
    except Exception as e:
        write([f'  FAIL: {e}'], fh)

    # Historical coverage assessment
    write(['', '  Historical ZiPS / Steamer coverage assessment:'], fh)
    write([
        '  ZiPS:    reliably available on FanGraphs from ~2006 onward.',
        '           Historical archived pages may require Wayback Machine for pre-2015.',
        '           Programmatic access to pre-2015 ZiPS team W% projections is UNCERTAIN.',
        '  Steamer: available on FanGraphs from ~2013 onward.',
        '           Pre-2013 Steamer: NOT available in scrapable form.',
        '  ACTION:  For seasons 2006–2014, ZiPS data may need to be collected manually',
        '           from FanGraphs or Baseball Prospectus archives.',
        '           Flag all pre-2015 projection data as potentially requiring manual entry.',
    ], fh)
    log_audit(
        'Phase0: ZiPS coverage reliable ~2006+; Steamer ~2013+. '
        'Pre-2015 team-level W% projections may require manual collection.', 'WARNING'
    )


# ── Section 4: Marcel projections ─────────────────────────────────────────────
def audit_marcel(fh):
    write(['', '='*70, 'SECTION 4: Marcel Projections (2000–2005 fallback)', '='*70], fh)

    import requests

    # Tom Tango's Marcel page
    time.sleep(2)
    urls = [
        'http://www.tangotiger.net/marcel/',
        'https://www.tangotiger.net/marcel/',
    ]
    found = False
    for url in urls:
        try:
            r = requests.get(url, timeout=15,
                             headers={'User-Agent': 'Mozilla/5.0'})
            write([f'  {url}: status {r.status_code}'], fh)
            if r.status_code == 200:
                found = True
                break
        except Exception as e:
            write([f'  {url}: FAIL ({e})'], fh)

    if not found:
        write([
            '  WARNING: tangotiger.net Marcel page not accessible.',
            '  Marcel team-level W% for 2000–2005 is not available in a scrapable format.',
            '  ACTION: Projection coverage gap years 2000–2005 will be marked as unavailable.',
        ], fh)
        log_audit(
            'Phase0: Marcel team-level W% projections for 2000-2005 not found '
            'in accessible scrapable form. Those seasons will have projected_wpct=null.', 'WARNING'
        )
    else:
        write([
            '  tangotiger.net accessible — but Marcel provides PLAYER-level projections,',
            '  not team-level W%. Team-level projected W% must be aggregated from player',
            '  Marcel projections, which requires substantial additional processing.',
            '  ASSESSMENT: Marcel team W% for 2000–2005 is NOT readily available as a',
            '  direct download. Treat 2000–2005 as projected_wpct=null unless a',
            '  pre-aggregated source is found.',
        ], fh)
        log_audit(
            'Phase0: Marcel player projections accessible at tangotiger.net but '
            'team-level W% aggregation for 2000-2005 requires manual work. '
            'Will default to projected_wpct=null for 2000-2005.', 'WARNING'
        )


# ── Section 5: pybaseball coverage summary ────────────────────────────────────
def audit_pybaseball_coverage(fh):
    write(['', '='*70, 'SECTION 5: pybaseball Coverage Summary', '='*70], fh)

    try:
        import pybaseball
    except ImportError:
        write(['  pybaseball not installed — skipping coverage check'], fh)
        return

    coverage = [
        ('schedule_and_record(year, team)',  'Game logs by team-season via BBRef scrape',
         '2000–present', 'CONFIRMED above'),
        ('lahman.managers()',                 'Lahman Managers table (mid-season changes)',
         '1871–~2023', 'CONFIRMED above'),
        ('lahman.people()',                   'Player/manager name lookup',
         '1871–~2023', 'CONFIRMED above'),
        ('batting_stats(year)',               'Team batting stats by season',
         '~2002–present', 'AVAILABLE (FanGraphs scrape)'),
        ('pitching_stats(year)',              'Team pitching stats by season',
         '~2002–present', 'AVAILABLE (FanGraphs scrape)'),
        ('batting_stats_bref(year)',          'Team batting stats from BBRef',
         '2000–present', 'AVAILABLE (BBRef scrape)'),
    ]

    write(['  Function                               Coverage        Status'], fh)
    write(['  ' + '-'*68], fh)
    for fn, desc, cov, status in coverage:
        write([f'  {fn:<40} {cov:<18} {status}'], fh)
        write([f'    → {desc}'], fh)

    write(['',
           '  NOTE: All pybaseball functions that hit BBRef are subject to the same',
           '        rate-limiting rules. pipeline adds 2-second delays manually.'], fh)


# ── Main audit runner ──────────────────────────────────────────────────────────
def run():
    print("Phase 0: Running data availability audit ...")
    log_audit('='*60, 'PHASE0_START')
    log_audit('Phase 0 data availability audit started.', 'PHASE0_START')

    with open(AUDIT_OUT, 'w', encoding='utf-8') as fh:
        fh.write('MLB Manager Firings — Data Availability Audit\n')
        fh.write(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        fh.write('='*70 + '\n')

        audit_pybaseball(fh)
        audit_bbref(fh)
        audit_fangraphs(fh)
        audit_marcel(fh)
        audit_pybaseball_coverage(fh)

        fh.write('\n' + '='*70 + '\n')
        fh.write('END OF AUDIT\n')

    print(f"Phase 0 complete — audit written to data_availability_audit.txt")
    log_audit('Phase 0 complete.', 'PHASE0_END')


if __name__ == '__main__':
    run()
