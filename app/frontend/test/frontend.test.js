import test from 'node:test'
import assert from 'node:assert/strict'

import { fetchHealth, fetchModelInfo, fetchScore } from '../src/api.js'
import { DEFAULT_MODEL_INFO } from '../src/config.js'
import { mockScore, mockTrace } from '../src/mockData.js'
import { featureDisplayName, isHighRisk, riskLevel } from '../src/riskUtils.js'


test('risk thresholds are consistent at boundaries', () => {
  assert.equal(riskLevel(39).key, 'low')
  assert.equal(riskLevel(40).key, 'mid')
  assert.equal(riskLevel(70).key, 'high')
  assert.equal(isHighRisk(69), false)
  assert.equal(isHighRisk(70), true)
})


test('anonymous Elliptic features are grouped for human review', () => {
  assert.equal(featureDisplayName('feat_1'), '거래 자체 특성 2')
  assert.equal(featureDisplayName('feat_124'), '연결 이웃 집계 특성 32')
  assert.equal(featureDisplayName('거래소 패턴'), '거래소 패턴')
})


test('default model evidence documents graph-feature contribution', () => {
  assert.equal(DEFAULT_MODEL_INFO.localFeatureCount, 93)
  assert.equal(DEFAULT_MODEL_INFO.neighborAggregateFeatureCount, 72)
  assert.ok(DEFAULT_MODEL_INFO.graphF1Lift > 0.06)
  assert.ok(DEFAULT_MODEL_INFO.graphEnhancedF1 > DEFAULT_MODEL_INFO.localOnlyF1)
})


test('frontend mock data is deterministic and path-safe', () => {
  assert.deepEqual(mockScore('232629023'), mockScore('232629023'))
  const trace = mockTrace('232629023', 2)
  const ids = new Set(trace.nodes.map((node) => node.id))
  assert.ok(trace.paths.length > 0)
  assert.ok(trace.paths.every((path) => path[0] === '232629023'))
  assert.ok(trace.paths.flat().every((id) => ids.has(id)))
})


test('API errors expose the backend detail instead of a generic status', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: '해당 txId가 없습니다.' }),
    { status: 404, headers: { 'Content-Type': 'application/json' } },
  )
  try {
    await assert.rejects(fetchScore('999999999'), /해당 txId가 없습니다/)
  } finally {
    globalThis.fetch = originalFetch
  }
})


test('health response is parsed', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => Response.json({ status: 'ok', modelLoaded: true, mode: 'model' })
  try {
    assert.deepEqual(await fetchHealth(), { status: 'ok', modelLoaded: true, mode: 'model' })
  } finally {
    globalThis.fetch = originalFetch
  }
})


test('model evidence response is parsed', async () => {
  const originalFetch = globalThis.fetch
  const evidence = {
    dataset: 'Elliptic Bitcoin',
    transactions: 203769,
    testSamples: 16670,
    illicitF1: 0.8051,
  }
  globalThis.fetch = async () => Response.json(evidence)
  try {
    assert.deepEqual(await fetchModelInfo(), evidence)
  } finally {
    globalThis.fetch = originalFetch
  }
})
