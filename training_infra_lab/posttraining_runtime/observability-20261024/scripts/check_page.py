import json
import os
from urllib.parse import urlsplit, unquote
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'.validation/browsers')
errors=[];requests=[];report={}
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu','--no-zygote'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('requestfailed',lambda r:requests.append({'url':r.url,'failure':r.failure}))
    page.goto((ROOT/'observability-20261024.html').resolve().as_uri())
    page.wait_for_selector('#chart-activity svg')
    page.wait_for_selector('#chart-nsys-trace svg')
    page.evaluate('document.fonts.ready')
    assert 'done' in page.locator('#diagnostic-badge').inner_text()
    assert 'passed=false' in page.locator('#diagnostic-quality').inner_text()
    assert '没有' in page.locator('#ncu-result').inner_text()
    assert 'No kernels were profiled' in page.locator('#ncu-result').inner_text()
    assert '18.73 GiB' in page.locator('#nsys-result').inner_text()
    assert '16.62 s' in page.locator('#nsys-result').inner_text()
    kernel_count=page.locator('#kernel-rows tr').count()
    page.fill('#kernel-search','nvjet')
    matched=page.locator('#kernel-rows tr').all_inner_texts()
    assert matched and all('nvjet' in text for text in matched)
    page.fill('#kernel-search','__no_matching_kernel__')
    assert page.locator('#kernel-rows tr').count()==0
    page.fill('#kernel-search','')
    assert page.locator('#kernel-rows tr').count()==kernel_count
    missing=[];checked=0
    for href in page.locator('a[href]').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))'):
        parsed=urlsplit(href)
        if not parsed.scheme and parsed.path:
            checked+=1
            if not (ROOT/unquote(parsed.path)).is_file():missing.append(href)
    assert not missing,missing
    report['downloads']={'local_links_checked':checked,'missing':missing}
    report['interaction']={'kernel_filter_verified':True,'kernel_rows':kernel_count,
                           'quality_and_capture_boundaries_visible':True}
    report['zoom']=page.evaluate('''() => {
      const chart=echarts.getInstanceByDom(document.getElementById('chart-nsys-bins'));
      chart.dispatchAction({type:'dataZoom',start:25,end:75});
      const option=chart.getOption().dataZoom[0];
      chart.dispatchAction({type:'dataZoom',start:0,end:100});
      return {start:option.start,end:option.end};
    }''')
    assert report['zoom']=={'start':25,'end':75}
    report['desktop']=page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth,charts:document.querySelectorAll(".chart svg").length,rows:document.querySelectorAll("#phase-rows tr").length})')
    assert report['desktop']['scroll']<=report['desktop']['width']
    assert report['desktop']['rows']==3
    assert report['desktop']['charts']==9
    page.select_option('#card-metric','memory.used [MiB]')
    assert '显存' in page.locator('#card-title').inner_text()
    page.evaluate('document.activeElement.blur(); window.scrollTo(0,0)')
    page.wait_for_timeout(100)
    page.screenshot(path=str(ROOT/'evidence/desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(250)
    report['mobile']=page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth,charts:document.querySelectorAll(".chart svg").length})')
    assert report['mobile']['scroll']<=report['mobile']['width']
    page.screenshot(path=str(ROOT/'evidence/mobile.png'),full_page=True)
    page.emulate_media(media='print')
    page.wait_for_timeout(100)
    report['print']=page.evaluate('({headerVisible:getComputedStyle(document.querySelector("header")).display!=="none",bodyWidth:document.body.scrollWidth})')
    assert not report['print']['headerVisible']
    browser.close()
report.update({'javascript_errors':errors,'failed_requests':requests,'passed':not errors and not requests})
(ROOT/'evidence/browser-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report['passed']
