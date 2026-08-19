import React, { useState, useCallback, useEffect } from 'react'
import InputBar from './components/InputBar'
import EvidenceBand from './components/EvidenceBand'
import CaseSummary from './components/CaseSummary'
import RiskPanel from './components/RiskPanel'
import GraphPanel from './components/GraphPanel'
import ReportPanel from './components/ReportPanel'
import { fetchHealth, fetchModelInfo, fetchScore, fetchTrace, fetchReport } from './api'
import { USE_MOCK, DEFAULT_HOPS, DEFAULT_MODEL_INFO } from './config'
import { isHighRisk } from './riskUtils'

export default function App() {
  const [txId, setTxId] = useState('')
  const [loading, setLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [error, setError] = useState('')
  const [backendMode, setBackendMode] = useState(USE_MOCK ? 'client-mock' : 'checking')
  const [modelInfo, setModelInfo] = useState(DEFAULT_MODEL_INFO)

  const [score, setScore] = useState(null) // /score 응답
  const [trace, setTrace] = useState(null) // /trace 응답
  const [report, setReport] = useState('') // /report 응답 텍스트

  useEffect(() => {
    let active = true
    fetchHealth()
      .then((health) => {
        if (active) setBackendMode(USE_MOCK ? 'client-mock' : health.mode)
      })
      .catch(() => {
        if (active) setBackendMode('offline')
      })
    fetchModelInfo()
      .then((info) => {
        if (active) setModelInfo(info)
      })
      .catch(() => {
        if (active) setModelInfo(DEFAULT_MODEL_INFO)
      })
    return () => {
      active = false
    }
  }, [])

  const analyze = useCallback(async (selectedId) => {
    const id = (typeof selectedId === 'string' ? selectedId : txId).trim()
    if (!id) return
    setTxId(id)
    setLoading(true)
    setError('')
    setScore(null)
    setTrace(null)
    setReport('')
    setReportLoading(false)
    let scoreRes
    let traceRes
    try {
      // 1) 위험 점수 + 2) 자금 흐름 그래프를 병렬 호출
      const [nextScore, nextTrace] = await Promise.all([
        fetchScore(id),
        fetchTrace(id, DEFAULT_HOPS),
      ])
      scoreRes = nextScore
      traceRes = nextTrace
      setScore(scoreRes)
      setTrace(traceRes)
    } catch (e) {
      setError(e.message || '분석 중 오류가 발생했습니다.')
      return
    } finally {
      setLoading(false)
    }

    // 3) 기본 분석 결과는 리포트 생성 실패와 무관하게 유지한다.
    const graphStats = {
      nodeCount: traceRes.nodes.length,
      edgeCount: traceRes.edges.length,
      highRiskCount: traceRes.nodes.filter((n) => isHighRisk(n.risk)).length,
      hops: DEFAULT_HOPS,
    }
    setReportLoading(true)
    try {
      const reportRes = await fetchReport(
        id,
        scoreRes.riskScore,
        scoreRes.label,
        scoreRes.topFactors,
        graphStats,
      )
      setReport(reportRes.report)
    } catch (e) {
      setError(`위험 분석은 완료됐지만 리포트 생성에 실패했습니다. ${e.message || ''}`)
    } finally {
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
            <div className="brand-sub">가상자산 자금세탁 탐지 및 추적 시스템</div>
          </div>
        </div>
        <div
          className={`mode-tag ${backendMode === 'model' ? 'live' : backendMode === 'offline' ? 'offline' : 'mock'}`}
          title="현재 분석 데이터 공급자"
        >
          {backendMode === 'model'
            ? 'MODEL API'
            : backendMode === 'offline'
              ? 'API OFFLINE'
              : backendMode === 'checking'
                ? 'API 확인 중'
                : backendMode === 'client-mock'
                  ? '브라우저 데모 모드'
                  : '서버 MOCK 모드'}
        </div>
      </header>

      <EvidenceBand info={modelInfo} />

      <InputBar
        value={txId}
        onChange={setTxId}
        onAnalyze={analyze}
        onExample={analyze}
        loading={loading}
      />

      {error && <div className="error-bar" role="alert"><strong>확인 필요</strong>{error}</div>}

      <CaseSummary txId={txId} score={score} trace={trace} />

      <main className="grid">
        <div className="col-left">
          <RiskPanel result={score} loading={loading} modelInfo={modelInfo} />
        </div>
        <div className="col-center">
          <GraphPanel trace={trace} loading={loading} />
        </div>
        <div className="col-right">
          <ReportPanel report={report} loading={reportLoading} />
        </div>
      </main>

      <footer className="app-footer">
        2026 금융 AI Challenge | ChainEye 방어적 분석 데모
      </footer>
    </div>
  )
}
