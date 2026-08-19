// =====================================================================
// ChainEye 프론트엔드 설정 (config)
// ---------------------------------------------------------------------
// 모든 값은 Vite 빌드타임 환경변수(import.meta.env.VITE_*)로 주입됩니다.
//
// API_BASE : FastAPI 백엔드의 베이스 URL.
//   - 기본값 '' (빈 문자열) = SAME ORIGIN(동일 출처). 백엔드가 빌드된 프론트를
//     정적 파일로 서빙하면 `POST /score` 는 배포 도메인 기준으로 해석됩니다.
//   - 별도 백엔드로 로컬 개발 시 VITE_API_BASE=http://localhost:8000 처럼 지정.
//
// USE_MOCK : true  -> 백엔드 없이 하드코딩된 샘플 데이터로 동작 (독립 데모)
//            false -> API_BASE 의 실제 백엔드를 호출 (프로덕션 기본값)
//   - 문자열 'true' 일 때만 mock. 기본값은 false(실제 API).
//   - .env.development 에서 VITE_USE_MOCK=true 로 두어 `npm run dev` 는
//     기본적으로 독립 mock 데모로 동작합니다.
// =====================================================================

// 빈 문자열('')이면 동일 출처(상대경로) 요청. 절대 URL이면 해당 호스트로 요청.
const env = import.meta.env ?? {}

export const API_BASE = env.VITE_API_BASE ?? ''

// 'true' 문자열일 때만 mock 모드. 그 외(미설정 포함)는 실제 백엔드 호출.
export const USE_MOCK = (env.VITE_USE_MOCK ?? 'false') === 'true'

// 트레이스(자금 흐름 그래프) 조회 시 탐색할 홉(hop) 수
export const DEFAULT_HOPS = 2

// 입력창에 노출할 예시 트랜잭션 ID (클릭 시 자동 입력)
export const EXAMPLE_CASES = [
  { txId: '232629023', label: '고위험 사례', description: '불법 라벨 거래' },
  { txId: '232438397', label: '정상 사례', description: '정상 라벨 거래' },
  { txId: '230425980', label: '미분류 사례', description: '라벨 없는 거래' },
]

export const EXAMPLE_TXIDS = EXAMPLE_CASES.map((example) => example.txId)

export const DEFAULT_MODEL_INFO = {
  active: false,
  activeModel: 'LightGBM with graph-neighbor aggregates',
  dataset: 'Elliptic Bitcoin',
  transactions: 203769,
  trainSamples: 29894,
  testSamples: 16670,
  illicitF1: 0.8050761421319796,
  illicitPrecision: 0.8940248027057497,
  illicitRecall: 0.7322253000923361,
  prAuc: 0.7994953163485768,
  rocAuc: 0.9317122177423582,
  decisionThreshold: 0.523809503088202,
  validationProtocol: 'rolling temporal validation and untouched future test',
  featureCount: 165,
  localFeatureCount: 93,
  neighborAggregateFeatureCount: 72,
  localOnlyF1: 0.743988684582744,
  graphEnhancedF1: 0.8050761421319796,
  graphF1Lift: 0.06108745754923561,
  graphPrAucLift: 0.013677003330201498,
}
