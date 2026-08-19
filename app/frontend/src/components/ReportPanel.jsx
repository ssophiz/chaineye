import React, { useState } from 'react'

// 아주 가벼운 마크다운-유사 렌더러 (외부 라이브러리 없이).
// # 제목, ## 소제목, - 목록, > 인용, **굵게** 정도만 처리.
function renderInline(text, keyPrefix) {
  // **굵게** 처리
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((p, i) => {
    if (/^\*\*[^*]+\*\*$/.test(p)) {
      return <strong key={`${keyPrefix}-b${i}`}>{p.slice(2, -2)}</strong>
    }
    if (/^`[^`]+`$/.test(p)) {
      return (
        <code key={`${keyPrefix}-c${i}`} className="inline-code">
          {p.slice(1, -1)}
        </code>
      )
    }
    return <span key={`${keyPrefix}-t${i}`}>{p}</span>
  })
}

function renderMarkdown(md) {
  const lines = md.replace(/[—–]/g, '-').split('\n')
  const out = []
  lines.forEach((line, idx) => {
    const key = `l${idx}`
    if (line.startsWith('# ')) {
      out.push(<h3 key={key} className="md-h1">{renderInline(line.slice(2), key)}</h3>)
    } else if (line.startsWith('## ')) {
      out.push(<h4 key={key} className="md-h2">{renderInline(line.slice(3), key)}</h4>)
    } else if (line.startsWith('> ')) {
      out.push(<blockquote key={key} className="md-quote">{renderInline(line.slice(2), key)}</blockquote>)
    } else if (/^\d+\.\s/.test(line)) {
      out.push(<div key={key} className="md-li ol">{renderInline(line, key)}</div>)
    } else if (line.startsWith('- ')) {
      out.push(<div key={key} className="md-li">• {renderInline(line.slice(2), key)}</div>)
    } else if (line.trim() === '') {
      out.push(<div key={key} className="md-gap" />)
    } else {
      out.push(<p key={key} className="md-p">{renderInline(line, key)}</p>)
    }
  })
  return out
}

export default function ReportPanel({ report, loading }) {
  const [copied, setCopied] = useState(false)

  const copyReport = async () => {
    if (!navigator.clipboard) return
    try {
      await navigator.clipboard.writeText(report)
      setCopied(true)
    } catch {
      setCopied(false)
      return
    }
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="panel report-panel">
      <div className="report-head">
        <div>
          <h2 className="panel-title">조사 보고서 초안</h2>
          <span className="panel-context">모델 수치 기반 | 담당자 검토 필요</span>
        </div>
        {report && (
          <button
            className="copy-btn"
            type="button"
            onClick={copyReport}
            title="리포트 원문 복사"
          >
            {copied ? '복사됨' : '복사'}
          </button>
        )}
      </div>
      {loading ? (
        <div className="report-skeleton" role="status" aria-busy="true">
          <div className="skeleton skeleton-line wide" />
          <div className="skeleton skeleton-line" />
          <div className="skeleton skeleton-line short" />
          <div className="skeleton skeleton-block" />
          <span className="sr-only">조사 보고서 초안을 생성하고 있습니다.</span>
        </div>
      ) : !report ? (
        <div className="empty">
          <strong>보고서 생성 대기</strong>
          <span>분석 결과가 나오면 근거, 경로, 후속 조치를 자동으로 정리합니다.</span>
        </div>
      ) : (
        <div className="report-body">{renderMarkdown(report)}</div>
      )}
    </div>
  )
}
