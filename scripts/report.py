import json
import base64
from pathlib import Path
from datetime import datetime

def img_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

with open('outputs/metrics.json') as f:
    metrics = json.load(f)

imgs = {
    'eda_trend':      'outputs/eda_01_weekly_trend.png',
    'eda_store_type': 'outputs/eda_02_store_type.png',
    'eda_season':     'outputs/eda_03_season.png',
    'eda_holiday':    'outputs/eda_04_holiday.png',
    'eda_corr':       'outputs/eda_05_correlation.png',
    'eda_dept':       'outputs/eda_06_top_departments.png',
    'fi':             'outputs/model_01_feature_importance.png',
    'pred':           'outputs/model_02_actual_vs_pred.png',
    'resid':          'outputs/model_03_residuals.png',
}
b64 = {k: img_b64(v) for k, v in imgs.items() if Path(v).exists()}

now = datetime.now().strftime('%Y-%m-%d %H:%M')

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>매출 예측 분석 리포트</title>
<style>
  :root {{
    --blue: #2563EB; --purple: #7C3AED; --red: #DC2626;
    --green: #16A34A; --gray: #6B7280; --bg: #F8FAFC; --card: #FFFFFF;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: #1E293B; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, var(--blue) 0%, var(--purple) 100%);
             color: white; padding: 48px 40px; }}
  .header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
  .header p {{ opacity: 0.85; font-size: 0.95rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}
  .section {{ margin-bottom: 48px; }}
  .section h2 {{ font-size: 1.4rem; font-weight: 700; color: #0F172A;
                 border-left: 4px solid var(--blue); padding-left: 12px; margin-bottom: 24px; }}
  .section h3 {{ font-size: 1.05rem; font-weight: 600; color: #334155; margin: 20px 0 12px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
  .metric-card {{ background: var(--card); border-radius: 12px; padding: 24px;
                  box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }}
  .metric-card .label {{ font-size: 0.78rem; color: var(--gray); text-transform: uppercase;
                          letter-spacing: .05em; margin-bottom: 8px; }}
  .metric-card .value {{ font-size: 1.9rem; font-weight: 700; }}
  .metric-card.blue .value {{ color: var(--blue); }}
  .metric-card.green .value {{ color: var(--green); }}
  .metric-card.purple .value {{ color: var(--purple); }}
  .metric-card.red .value {{ color: var(--red); }}
  .img-card {{ background: var(--card); border-radius: 12px; padding: 24px;
               box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px; }}
  .img-card img {{ width: 100%; height: auto; border-radius: 6px; }}
  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .img-grid .img-card {{ margin-bottom: 0; }}
  .finding-list {{ background: var(--card); border-radius: 12px; padding: 24px;
                   box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .finding-list li {{ padding: 10px 0; border-bottom: 1px solid #F1F5F9;
                      display: flex; align-items: flex-start; gap: 10px; }}
  .finding-list li:last-child {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.72rem; font-weight: 600; white-space: nowrap; }}
  .badge-blue {{ background: #DBEAFE; color: var(--blue); }}
  .badge-green {{ background: #DCFCE7; color: var(--green); }}
  .badge-purple {{ background: #EDE9FE; color: var(--purple); }}
  .badge-orange {{ background: #FEF3C7; color: #D97706; }}
  .info-box {{ background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px;
               padding: 16px 20px; margin-bottom: 20px; font-size: 0.9rem; color: #1E40AF; }}
  .footer {{ text-align: center; padding: 32px; color: var(--gray); font-size: 0.82rem;
             border-top: 1px solid #E2E8F0; margin-top: 40px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Weekly Sales Forecasting Report</h1>
  <p>Dataset: Retail Store Sales Forecasting &nbsp;|&nbsp; Model: LightGBM &nbsp;|&nbsp; Generated: {now}</p>
</div>
<div class="container">

  <!-- 분석 개요 -->
  <div class="section">
    <h2>분석 개요</h2>
    <div class="info-box">
      <strong>목표:</strong> 50개 소매 매장, 20개 부서의 주간 매출(weekly_sales)을 시계열 피처 기반으로 예측<br>
      <strong>데이터:</strong> 2022-01-01 ~ 2024-12-21 | 156,000행 | 훈련 131,000 / 테스트 12,000<br>
      <strong>피처:</strong> lag(1, 4, 13주), rolling mean/std, 시즌, 휴일, 마크다운, CPI, 실업률, 매장 특성
    </div>
  </div>

  <!-- 모델 성능 -->
  <div class="section">
    <h2>모델 성능 (LightGBM)</h2>
    <div class="metrics-grid">
      <div class="metric-card blue">
        <div class="label">RMSE</div>
        <div class="value">{metrics['rmse']:,.0f}</div>
      </div>
      <div class="metric-card purple">
        <div class="label">MAE</div>
        <div class="value">{metrics['mae']:,.0f}</div>
      </div>
      <div class="metric-card green">
        <div class="label">R²</div>
        <div class="value">{metrics['r2']:.3f}</div>
      </div>
      <div class="metric-card red">
        <div class="label">MAPE</div>
        <div class="value">{metrics['mape']:.1f}%</div>
      </div>
    </div>
  </div>

  <!-- 피처 중요도 & 예측 결과 -->
  <div class="section">
    <h2>모델 결과</h2>
    <div class="img-grid">
      {'<div class="img-card"><h3>Feature Importance (Top 15)</h3><img src="data:image/png;base64,' + b64["fi"] + '"></div>' if "fi" in b64 else ''}
      {'<div class="img-card"><h3>Actual vs Predicted — Store 1, Dept 1</h3><img src="data:image/png;base64,' + b64["pred"] + '"></div>' if "pred" in b64 else ''}
    </div>
    {'<div class="img-card" style="margin-top:20px"><h3>Residual Analysis</h3><img src="data:image/png;base64,' + b64["resid"] + '"></div>' if "resid" in b64 else ''}
  </div>

  <!-- EDA 결과 -->
  <div class="section">
    <h2>탐색적 데이터 분석 (EDA)</h2>

    <div class="img-card">
      <h3>Weekly Sales Trend (전체 합산)</h3>
      {'<img src="data:image/png;base64,' + b64["eda_trend"] + '">' if "eda_trend" in b64 else ''}
    </div>

    <div class="img-grid">
      <div class="img-card">
        <h3>Store Type별 평균 매출</h3>
        {'<img src="data:image/png;base64,' + b64["eda_store_type"] + '">' if "eda_store_type" in b64 else ''}
      </div>
      <div class="img-card">
        <h3>Season별 평균 매출</h3>
        {'<img src="data:image/png;base64,' + b64["eda_season"] + '">' if "eda_season" in b64 else ''}
      </div>
      <div class="img-card">
        <h3>휴일 여부별 평균 매출</h3>
        {'<img src="data:image/png;base64,' + b64["eda_holiday"] + '">' if "eda_holiday" in b64 else ''}
      </div>
      <div class="img-card">
        <h3>피처 상관관계 히트맵</h3>
        {'<img src="data:image/png;base64,' + b64["eda_corr"] + '">' if "eda_corr" in b64 else ''}
      </div>
    </div>

    <div class="img-card" style="margin-top:20px">
      <h3>부서별 평균 주간 매출 (Top 20)</h3>
      {'<img src="data:image/png;base64,' + b64["eda_dept"] + '">' if "eda_dept" in b64 else ''}
    </div>
  </div>

  <!-- 핵심 인사이트 -->
  <div class="section">
    <h2>핵심 인사이트</h2>
    <div class="finding-list">
      <ul style="list-style:none">
        <li>
          <span class="badge badge-blue">EDA #1</span>
          <span><strong>Fall 시즌이 최고 매출</strong> — 예상과 달리 Winter가 아닌 Fall(가을)에 피크. 추수감사절·블랙프라이데이 효과로 추정.</span>
        </li>
        <li>
          <span class="badge badge-green">EDA #2</span>
          <span><strong>휴일 매출 &gt; 비휴일</strong> — 공휴일 프로모션 효과가 매출에 유의미하게 기여.</span>
        </li>
        <li>
          <span class="badge badge-purple">EDA #3</span>
          <span><strong>매장 타입 A &gt; B &gt; C</strong> — 대형 매장(Type A)이 소형 대비 압도적 매출 우위. store_size와 강한 양의 상관.</span>
        </li>
        <li>
          <span class="badge badge-orange">Model #1</span>
          <span><strong>lag_1 피처가 가장 중요</strong> — 직전 주 매출이 예측력을 가장 많이 기여. 단기 관성이 지배적.</span>
        </li>
        <li>
          <span class="badge badge-blue">Model #2</span>
          <span><strong>R² = {metrics['r2']:.3f}</strong> — 모델이 분산의 약 {int(metrics['r2']*100)}%를 설명. 부서별 분포 편차(MAPE {metrics['mape']:.0f}%)가 높아 소량 부서 예측 정확도 개선 필요.</span>
        </li>
      </ul>
    </div>
  </div>

</div>
<div class="footer">
  Generated by Claude Code &nbsp;·&nbsp; {now} &nbsp;·&nbsp; LightGBM v4
</div>
</body>
</html>"""

out = Path('outputs/report.html')
out.write_text(html, encoding='utf-8')
print(f"리포트 저장: {out.resolve()}")
