import React from 'react'
import { featureDisplayName, riskLevel } from '../riskUtils'

// 원형 게이지 (순수 SVG, 외부 라이브러리 불필요)
function Gauge({ score }) {
  const level = riskLevel(score)
  const size = 190
  const stroke = 16
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(100, score)) / 100
  const dash = c * pct

  return (
    <div className="gauge-wrap">
      <svg width={size} height={size} className="gauge">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#232a3a"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={level.color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c - dash}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dasharray 0.6s ease, stroke 0.3s' }}
        />
      </svg>
      <div className="gauge-center">
        <div className="gauge-score" style={{ color: level.color }}>
          {score}
        </div>
        <div className="gauge-max">/ 100</div>
      </div>
    </div>
  )
}

export default function RiskPanel({ result, loading, modelInfo }) {
  if (loading) {
    return (
      <div className="panel risk-panel" aria-busy="true">
        <h2 className="panel-title">위험도 평가</h2>
        <div className="skeleton skeleton-gauge" />
        <div className="skeleton skeleton-line wide" />
        <div className="skeleton skeleton-line" />
        <span className="sr-only">위험 점수를 계산하고 있습니다.</span>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="panel risk-panel">
        <h2 className="panel-title">위험도 평가</h2>
        <div className="empty">
          <strong>판정 대기</strong>
          <span>고위험 사례를 선택하면 모델 점수와 판단 근거를 확인할 수 있습니다.</span>
        </div>
      </div>
    )
  }

  const { riskScore, label, topFactors } = result
  const level = riskLevel(riskScore)
  const factors = topFactors || []
  const maxImpact = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.0001)

  return (
    <div className="panel risk-panel">
      <h2 className="panel-title">위험도 평가</h2>

      <Gauge score={riskScore} />

      <div className="risk-badge" style={{ background: level.color }}>
        {level.label}
        <span className="risk-badge-sub">
          {label === 'illicit' ? '불법 자금 의심' : '정상 거래 추정'}
        </span>
      </div>

      <div className="decision-note">
        <span>모델 판정 임계값</span>
        <strong>{((modelInfo?.decisionThreshold ?? 0.5238) * 100).toFixed(1)}점</strong>
      </div>

      <div className="factors">
        <div className="factors-title">핵심 위험 근거</div>
        <ul className="factors-list">
          {factors.map((f, i) => {
            const positive = f.impact >= 0
            return (
              <li key={`${f.feature}-${i}`} className="factor-item">
                <span
                  className="factor-bar"
                  style={{
                    width: `${Math.min(100, (Math.abs(f.impact) / maxImpact) * 100)}%`,
                    background: positive ? '#ef4444' : '#22c55e',
                  }}
                />
                <span className="factor-text" title={`원본 피처: ${f.feature}`}>
                  <strong>{featureDisplayName(f.feature)}</strong>
                  <small>{positive ? '위험 증가' : '위험 완화'} | {f.feature}</small>
                </span>
                <span
                  className="factor-impact"
                  style={{ color: positive ? '#f87171' : '#4ade80' }}
                >
                  {positive ? '+' : ''}
                  {Number(f.impact).toFixed(3)}
                </span>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
