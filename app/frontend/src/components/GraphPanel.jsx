import React, { useMemo } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import { isHighRisk } from '../riskUtils'

// paths(의심 자금세탁 경로)에 포함된 방향성 간선 집합을 "source->target" 키로 생성
function buildPathEdgeSet(trace) {
  const set = new Set()
  const paths = (trace && trace.paths) || []
  for (const path of paths) {
    for (let i = 0; i + 1 < path.length; i++) {
      set.add(`${path[i]}->${path[i + 1]}`)
    }
  }
  return set
}

// trace 응답 -> cytoscape elements 변환
function buildElements(trace) {
  if (!trace) return []
  const pathEdges = buildPathEdgeSet(trace)
  const nodes = trace.nodes.map((n) => {
    // 백엔드 illicit 플래그 우선, 없으면 risk 기준으로 판정(하위 호환)
    const highRisk = n.illicit != null ? n.illicit : isHighRisk(n.risk)
    let cls = 'normal'
    if (n.focus) cls = 'focus'
    else if (highRisk) cls = 'high'
    return {
      data: {
        id: n.id,
        label: n.id.length > 10 ? `${n.id.slice(0, 6)}…` : n.id,
        risk: n.risk,
        cls,
      },
      classes: cls,
    }
  })
  const edges = trace.edges.map((e, i) => {
    const onPath = pathEdges.has(`${e.source}->${e.target}`)
    return {
      data: {
        id: `e${i}-${e.source}-${e.target}`,
        source: e.source,
        target: e.target,
        laundering: onPath,
      },
      classes: onPath ? 'laundering' : '',
    }
  })
  return [...nodes, ...edges]
}

const stylesheet = [
  {
    selector: 'node',
    style: {
      'background-color': '#6b7280', // gray - 정상
      label: 'data(label)',
      color: '#cbd5e1',
      'font-size': '9px',
      'text-valign': 'bottom',
      'text-halign': 'center',
      'text-margin-y': 4,
      width: 34,
      height: 34,
      'border-width': 2,
      'border-color': '#1f2430',
    },
  },
  {
    selector: 'node.high',
    style: {
      'background-color': '#ef4444', // red - 고위험
      'border-color': '#7f1d1d',
    },
  },
  {
    selector: 'node.focus',
    style: {
      'background-color': '#dc2626', // red - 포커스(대상)
      'border-color': '#fca5a5',
      'border-width': 4,
      width: 50,
      height: 50,
      'font-size': '11px',
      color: '#fff',
      'font-weight': 'bold',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 2,
      'line-color': '#3f4a5f',
      'target-arrow-color': '#3f4a5f',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'arrow-scale': 1.1,
    },
  },
  {
    // 의심 자금세탁 경로에 속한 간선: 밝은 주황/빨강, 굵고 강조
    selector: 'edge.laundering',
    style: {
      width: 5,
      'line-color': '#f97316', // bright orange - 경보색
      'target-arrow-color': '#f97316',
      'arrow-scale': 1.5,
      'line-style': 'solid',
      'z-index': 20,
      'overlay-color': '#f97316',
      'overlay-opacity': 0.12,
      'overlay-padding': 3,
      // 흐름 방향으로 흐르는 "marching ants" 점선 (offset은 아래 rAF 루프가 구동)
      'line-dash-pattern': [10, 6],
      'line-dash-offset': 0,
    },
  },
]

const layout = {
  name: 'breadthfirst',
  directed: true,
  spacingFactor: 1.3,
  padding: 24,
  animate: false,
}

export default function GraphPanel({ trace, loading }) {
  const elements = useMemo(() => buildElements(trace), [trace])
  const pathCount = (trace && trace.paths && trace.paths.length) || 0

  return (
    <div className={`panel graph-panel ${!trace ? 'is-empty' : ''}`}>
      <div className="graph-head">
        <h2 className="panel-title">자금 흐름 그래프</h2>
        {trace && (
          <div className="graph-stats" aria-label="그래프 요약">
            <span>거래 {trace.nodes.length}</span>
            <span>이동 {trace.edges.length}</span>
            <span>의심 경로 {pathCount}</span>
          </div>
        )}
      </div>
      {loading ? (
        <div className="graph-skeleton" aria-busy="true">
          <div className="skeleton-node n1" />
          <div className="skeleton-node n2" />
          <div className="skeleton-node n3" />
          <span className="sr-only">자금 흐름 그래프를 구성하고 있습니다.</span>
        </div>
      ) : !trace ? (
        <div className="empty">
          <strong>흐름 추적 대기</strong>
          <span>거래를 선택하면 2홉 이내 자금 이동과 고위험 연결이 표시됩니다.</span>
        </div>
      ) : (
        <>
          <div
            className="graph-canvas"
            role="img"
            aria-label={`트랜잭션 ${trace.nodes.length}개와 자금 이동 ${trace.edges.length}건의 방향성 그래프`}
          >
            <CytoscapeComponent
              key={trace.nodes.map((n) => n.id).join(',')}
              elements={elements}
              stylesheet={stylesheet}
              layout={layout}
              style={{ width: '100%', height: '100%' }}
              minZoom={0.3}
              maxZoom={2.5}
            />
          </div>
          <div className="graph-legend">
            <span>
              <i className="dot" style={{ background: '#dc2626' }} /> 대상 트랜잭션
            </span>
            <span>
              <i className="dot" style={{ background: '#ef4444' }} /> 고위험(70+)
            </span>
            <span>
              <i className="dot" style={{ background: '#6b7280' }} /> 일반
            </span>
            <span>
              <i
                className="dot"
                style={{ background: '#f97316', borderRadius: 2 }}
              />{' '}
              의심 자금세탁 경로
              {pathCount > 0 ? ` (${pathCount})` : ''}
            </span>
          </div>
        </>
      )}
    </div>
  )
}
