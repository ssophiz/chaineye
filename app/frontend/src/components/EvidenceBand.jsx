import React from 'react'

function Metric({ label, value, detail }) {
  return (
    <div className="evidence-metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
      <span>{detail}</span>
    </div>
  )
}

export default function EvidenceBand({ info }) {
  return (
    <section className="evidence-band" aria-labelledby="evidence-title">
      <div className="evidence-copy">
        <span className="evidence-kicker">검증된 모델 근거</span>
        <h1 id="evidence-title">미래 거래 구간에서도 불법 흐름을 가려냅니다</h1>
        <p>
          Elliptic 거래 {info.transactions.toLocaleString('ko-KR')}건을 시간 순서대로
          검증하고, 테스트 구간은 모델 선택에 사용하지 않았습니다.
        </p>
        <div className="feature-proof" aria-label="라이브 모델 입력 특성 구성">
          <strong>라이브 LightGBM 입력</strong>
          <span>거래 자체 특성 {info.localFeatureCount ?? 93}개</span>
          <span aria-hidden="true">+</span>
          <span>그래프 이웃 집계 특성 {info.neighborAggregateFeatureCount ?? 72}개</span>
          <em>그래프 특성 포함 시 F1 +{(info.graphF1Lift ?? 0.061087).toFixed(3)}</em>
        </div>
      </div>
      <dl className="evidence-metrics">
        <Metric
          label="불법 거래 F1"
          value={info.illicitF1.toFixed(3)}
          detail={`미래 테스트 ${info.testSamples.toLocaleString('ko-KR')}건`}
        />
        <Metric
          label="정밀도"
          value={info.illicitPrecision.toFixed(3)}
          detail="오탐 94건"
        />
        <Metric
          label="PR-AUC"
          value={info.prAuc.toFixed(3)}
          detail="불균형 대표 지표"
        />
      </dl>
    </section>
  )
}
