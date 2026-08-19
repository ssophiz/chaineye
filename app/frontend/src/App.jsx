import React, { useState, useCallback, useEffect } from 'react'
import InputBar from './components/InputBar'
import RiskPanel from './components/RiskPanel'
import GraphPanel from './components/GraphPanel'
import ReportPanel from './components/ReportPanel'
import { fetchScore, fetchTrace, fetchReport, fetchHealth } from './api'
import { USE_MOCK, DEFAULT_HOPS } from './config'
import { isHighRisk } from './riskUtils'

export default function App() {
  const [txId, setTxId] = useState('')
  const [loading, setLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [error, setError] = useState('')
  const [providerMode, setProviderMode] = useState(
    USE_MOCK ? 'mock' : 'checking',
  )

  const [score, setScore] = useState(null) // /score 응답
  const [trace, setTrace] = useState(null) // /trace 응답
  const [report, setReport] = useState('') // /report 응답 텍스트
  const [reportGenerator, setReportGenerator] = useState('')

  useEffect(() => {
    let active = true
    let retryTimer

    const checkHealth = () => {
      fetchHealth()
        .then((health) => {
          if (!active) return
          setProviderMode(health.mode || 'unavailable')
          if (!health.modelLoaded) retryTimer = setTimeout(checkHealth, 3000)
        })
        .catch(() => {
          if (!active) return
          setProviderMode('unavailable')
          retryTimer = setTimeout(checkHealth, 3000)
        })
    }

    checkHealth()
    return () => {
      active = false
      clearTimeout(retryTimer)
    }
  }, [])

  const analyze = useCallback(async (requestedTxId) => {
    // 버튼 클릭 시에는 MouseEvent가 전달되고, 예시 클릭 시에는 txId 문자열이 전달된다.
    // 예시 값은 setState 반영을 기다리지 않고 즉시 분석에 사용한다.
    const id = (typeof requestedTxId === 'string' ? requestedTxId : txId).trim()
    if (!id) return
    setLoading(true)
    setError('')
    setReport('')
    setReportGenerator('')
    setReportLoading(true)
    try {
      // 1) 위험 점수 + 2) 자금 흐름 그래프를 병렬 호출
      const [scoreRes, traceRes] = await Promise.all([
        fetchScore(id),
        fetchTrace(id, DEFAULT_HOPS),
      ])
      setScore(scoreRes)
      setTrace(traceRes)

      if (scoreRes.label === 'unknown' || scoreRes.riskScore == null) {
        setReport(`# 분석 보류

대상 트랜잭션은 현재 모델의 데이터 범위에 없어 점수를 산출하지 않았습니다.

- **정상 거래로 판정한 것이 아닙니다.**
- 온체인 피처 수집 또는 별도 조사 후 다시 평가해야 합니다.
- 자동 보고서와 STR 권고는 근거가 확보될 때까지 생성하지 않습니다.`)
        setReportGenerator('suppressed')
        setReportLoading(false)
        return
      }

      // 3) 그래프 통계 계산 후 리포트 요청
      const graphStats = {
        nodeCount: traceRes.nodes.length,
        edgeCount: traceRes.edges.length,
        highRiskCount: traceRes.nodes.filter((n) =>
          n.modelPositive != null
            ? n.modelPositive
            : isHighRisk(n.risk, scoreRes.decisionThreshold),
        ).length,
        hops: DEFAULT_HOPS,
      }
      const reportRes = await fetchReport(
        id,
        scoreRes.riskScore,
        scoreRes.label,
        scoreRes.decisionThreshold,
        scoreRes.topFactors,
        graphStats,
      )
      setReport(reportRes.report)
      setReportGenerator(reportRes.generator || 'template')
    } catch (e) {
      setError(e.message || '분석 중 오류가 발생했습니다.')
      setScore(null)
      setTrace(null)
    } finally {
      setLoading(false)
      setReportLoading(false)
    }
  }, [txId])

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="logo">◈</span>
          <div>
            <div className="brand-name">
              ChainEye <span className="brand-ko">체인아이</span>
            </div>
            <div className="brand-sub">가상자산 AML 모델 검증 · 판단지원 시스템</div>
          </div>
        </div>
        <div className={`mode-tag ${providerMode}`}>
          {providerMode === 'model'
            ? 'MODEL API'
            : providerMode === 'mock'
              ? 'MOCK 데모 모드'
              : providerMode === 'checking'
                ? 'MODEL 확인 중'
                : 'MODEL 사용 불가'}
        </div>
      </header>

      <InputBar
        value={txId}
        onChange={setTxId}
        onAnalyze={analyze}
        onSelectExample={(exampleTxId) => {
          setTxId(exampleTxId)
          analyze(exampleTxId)
        }}
        loading={loading}
      />

      {error && <div className="error-bar">⚠ {error}</div>}

      <main className="grid">
        <div className="col-left">
          <RiskPanel result={score} />
        </div>
        <div className="col-center">
          <GraphPanel trace={trace} />
        </div>
        <div className="col-right">
          <ReportPanel
            report={report}
            generator={reportGenerator}
            loading={reportLoading}
          />
        </div>
      </main>

    </div>
  )
}
