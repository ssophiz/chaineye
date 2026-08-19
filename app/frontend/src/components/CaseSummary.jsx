import React from 'react'
import { isHighRisk } from '../riskUtils.js'

export default function CaseSummary({ txId, score, trace }) {
  if (!score || !trace) return null

  const highRiskCount = trace.nodes.filter((node) => isHighRisk(node.risk)).length
  const pathCount = trace.paths?.length ?? 0

  return (
    <section className="case-summary" aria-live="polite" aria-label="분석 결과 요약">
      <div className="case-identity">
        <span>분석 완료</span>
        <strong>{txId}</strong>
      </div>
      <dl>
        <div>
          <dt>모델 판정</dt>
          <dd className={score.label === 'illicit' ? 'danger' : 'safe'}>
            {score.label === 'illicit' ? '불법 자금 의심' : '정상 거래 추정'}
          </dd>
        </div>
        <div>
          <dt>확인 거래</dt>
          <dd>{trace.nodes.length}건</dd>
        </div>
        <div>
          <dt>자금 이동</dt>
          <dd>{trace.edges.length}건</dd>
        </div>
        <div>
          <dt>고위험 노드</dt>
          <dd>{highRiskCount}건</dd>
        </div>
        <div>
          <dt>의심 경로</dt>
          <dd>{pathCount}건</dd>
        </div>
      </dl>
    </section>
  )
}
