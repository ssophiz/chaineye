import React from 'react'
import { EXAMPLE_CASES } from '../config'

// 상단 입력바: txId 입력 + 분석 버튼 + 예시 칩
export default function InputBar({ value, onChange, onAnalyze, onExample, loading }) {
  const handleSubmit = (event) => {
    event.preventDefault()
    if (!loading && value.trim()) onAnalyze()
  }

  return (
    <div className="input-bar">
      <div className="input-intro">
        <strong>거래 ID로 위험 흐름 재현</strong>
        <span>아래 검증 사례를 누르면 점수, 자금 경로, 조사 초안이 한 번에 생성됩니다.</span>
      </div>
      <form className="input-row" onSubmit={handleSubmit}>
        <div className="input-field">
          <label htmlFor="tx-id">Elliptic 트랜잭션 ID</label>
          <input
            id="tx-id"
            className="tx-input"
            type="text"
            inputMode="numeric"
            pattern="[0-9]+"
            maxLength={32}
            autoComplete="off"
            placeholder="예: 232629023"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            spellCheck={false}
            aria-describedby="tx-id-help"
          />
          <span id="tx-id-help" className="field-help">공개 Elliptic 데이터셋의 숫자형 ID를 입력하세요.</span>
        </div>
        <button
          className="analyze-btn"
          type="submit"
          disabled={loading || !value.trim()}
        >
          {loading ? '분석 중...' : '분석 실행'}
        </button>
      </form>
      <div className="chips">
        <span className="chips-label">원클릭 검증</span>
        {EXAMPLE_CASES.map((example) => (
          <button
            key={example.txId}
            className="chip"
            type="button"
            title={`${example.description}: ${example.txId}`}
            onClick={() => onExample(example.txId)}
            disabled={loading}
          >
            <strong>{example.label}</strong>
            <span>{example.txId}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
