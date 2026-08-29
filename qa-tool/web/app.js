/** qa-tool web UI — mirrors qa-tool/draft.py heuristics (standalone, no app import). */

let universe = [];

function normalize(text) {
  return (text || "")
    .replace(/\s+/g, "")
    .replace(/[ㆍ·・‧]/g, "")
    .toLowerCase();
}

function detectCompany(query) {
  const compact = normalize(query);
  let best = null;
  for (const row of universe) {
    for (const label of [row.listed_name, row.corp_name]) {
      const key = normalize(label);
      if (key && compact.includes(key)) {
        if (!best || key.length > best.key.length) {
          best = { key, row };
        }
      }
    }
  }
  if (!best) {
    return { listed: "(회사명)", corp: "(corp_name)", inUniverse: false };
  }
  return {
    listed: best.row.listed_name,
    corp: best.row.corp_name,
    inUniverse: true,
  };
}

function detectDocGroup(query) {
  const compact = normalize(query);
  if (/소유상황보고서|대량보유|국민연금|보유비율|보유주식|변동후|변동전/.test(compact)) {
    return "holding";
  }
  const majorTerms = [
    "자기주식취득신탁계약해지",
    "자기주식취득신탁계약체결",
    "자기주식취득신탁",
    "자기주식취득",
    "자기주식처분",
    "자기주식소각",
    "유상증자",
    "전환사채",
    "회사분할",
    "흡수합병",
    "합병",
  ];
  if (majorTerms.some((t) => compact.includes(t))) return "major";
  if (/시설투자|신규시설|단일판매|공급계약|수주계약|투자판단/.test(compact)) return "exchange";
  if (compact.includes("계약해지") || compact.includes("계약해지금액")) {
    if (/자기주식|신탁계약|신탁/.test(compact)) return "major";
    return "exchange";
  }
  return "periodic";
}

function detectBasis(query) {
  if (/별도/.test(query)) return "별도";
  if (/연결/.test(query)) return "연결";
  if (/국민연금|보유|계약|시설/.test(query)) return "해당없음";
  return "(질문에 연결/별도 명시 권장)";
}

function detectReportHint(query) {
  if (/사업보고서/.test(query)) return "사업보고서 (연말 결산)";
  if (/반기보고서|상반기|하반기/.test(query)) return "반기보고서";
  const qm = query.match(/(20\d{2})\s*년?\s*([1-4])\s*분기/);
  if (qm) {
    const year = qm[1];
    const q = Number(qm[2]);
    return `분기보고서 (${year}.${String(q * 3).padStart(2, "0")}) · ${year}년 ${q}분기`;
  }
  const ym = query.match(/(20\d{2})\s*년/);
  if (ym) return `${ym[1]}년 (보고서 종류 명시 필요)`;
  return "(기간·보고서 종류 명시)";
}

function detectTaskType(query, docGroup) {
  const compact = normalize(query);
  if (docGroup === "holding") return "holding_change";
  if (docGroup === "major") {
    if (compact.includes("자기주식취득신탁계약해지") || (compact.includes("신탁") && compact.includes("해지"))) {
      return "corporate_event · treasury_share_trust_termination";
    }
    if (compact.includes("자기주식처분")) return "corporate_event · treasury_share_disposal";
    if (compact.includes("자기주식취득")) return "corporate_event · treasury_share_acquisition";
    return "major_event";
  }
  if (docGroup === "exchange") {
    if (/시설|투자\s*종료|투자목적|자기자본/.test(query)) return "facility_investment";
    return "supply_contract / exchange_event";
  }
  if (/상장일/.test(query)) return "listing_history";
  if (/구성|내역|수익\s*구분|재화|용역/.test(query)) return "periodic_fact · metric_view=breakdown";
  if (/매출|영업이익|당기순|자산|부채|자본|영업외|금융비용|EPS|주당/.test(query)) {
    return "periodic_fact · financial_metric";
  }
  return "periodic_fact / general_evidence";
}

function buildMustInclude(query, docGroup, basis) {
  const lines = [];
  if (basis === "연결" || basis === "별도") lines.push(`재무제표 기준: ${basis}`);
  if (docGroup === "periodic" && /구성|내역/.test(query)) {
    lines.push("섹션: 고객과의 계약에서 생기는 수익의 구분 (또는 동의 표)");
    lines.push("행: 재화·용역·로열티·금융·건설·기타 (공시에 있는 줄만)");
  } else if (docGroup === "periodic" && /매출|영업|순이익|자산|부채|자본/.test(query)) {
    lines.push("손익/재무상태표 해당 과목 한 줄 (질문한 지표만)");
  } else if (docGroup === "major" && /신탁.*해지|자기주식.*소각/.test(query)) {
    lines.push("report_nm: 주요사항보고서(자기주식취득신탁계약해지결정) 등");
    lines.push("신탁계약 해지·소각(예정) 관련 항목 (공시에 있는 경우만)");
  } else if (docGroup === "exchange" && /시설|투자/.test(query)) {
    lines.push("투자금액, 자기자본 대비(%), 투자목적, 시작·종료일, 이사회 결의일");
  } else if (docGroup === "exchange") {
    lines.push("계약상대, 계약금액, 계약기간, 최근매출 대비(%)");
  } else if (docGroup === "holding") {
    lines.push("보고자, 변동일, 변동 전·후 주식수·보유비율");
  }
  lines.push("숫자 단위 (백만원/주/% 등 공시 그대로)");
  return lines;
}

function buildMustNot(docGroup) {
  const common = "코퍼스 밖 공시·뉴스, 전년 대비·성장성 해석, 표에 없는 재분류";
  if (docGroup === "periodic") return `${common}, 질문과 다른 과목(매출원가·EPS 등) 혼입`;
  if (docGroup === "exchange") return `${common}, 현금조달 가능 여부 등 공시에 없는 판단`;
  if (docGroup === "major") return `${common}, exchange(공급계약 해지)로 오인`;
  return common;
}

function buildNegative(query, docGroup) {
  const compact = normalize(query);
  if (docGroup === "periodic" && /구성|내역/.test(query)) {
    return "손익계산서 매출액 총액 한 줄만 답함 (구성 질문의 대표 오답)";
  }
  if (docGroup === "major" && compact.includes("신탁") && compact.includes("해지")) {
    return "doc_group=exchange, subtype=단일판매공급계약해지 (신탁계약해지의 계약해지 부분문자열)";
  }
  if (docGroup === "exchange") return "정정 전 숫자·종료일 (정정본 존재 시)";
  if (docGroup === "holding") return "피투자회사와 보고자 혼동, 정기공시 재무표와 혼합";
  return "(공시 확인 후 기록)";
}

function buildEvidence(docGroup) {
  const prefix = { periodic: "periodic_", exchange: "exchange_", holding: "holding_", major: "major_" }[docGroup];
  return [
    `doc_id: ${prefix}(rcept_no)`,
    "section_path: (표 제목)",
    "manifest: data/corpus/manifest.jsonl에서 corp_name·rcept_dt 검색",
  ];
}

function buildQuestionId(question, corp, listed, prefix, index) {
  if (prefix) return `${prefix}${String(index || 1).padStart(2, "0")}`;
  const slugSource = corp !== "(corp_name)" ? corp : listed;
  const slug = slugSource.replace(/[^0-9A-Za-z가-힣]/g, "").slice(0, 4).toUpperCase() || "QA";
  const seq = Math.max(1, (question.length % 9) + 1);
  return `${slug}${String(seq).padStart(2, "0")}`;
}

function buildDraft(question, prefix, index) {
  const text = question.trim();
  if (!text) return null;
  const { listed, corp, inUniverse } = detectCompany(text);
  const docGroup = detectDocGroup(text);
  const basis = detectBasis(text);
  const mustInclude = buildMustInclude(text, docGroup, basis);
  const notes = /정정/.test(text)
    ? "정정본 우선 · 정정 전후 diff 확인"
    : "(공시 열람 후 doc_id·기대답 숫자 보완)";
  return {
    question_id: buildQuestionId(text, corp, listed, prefix, index),
    question: text,
    listed_name: listed,
    corp_name: corp,
    doc_group: docGroup,
    report_hint: detectReportHint(text),
    basis,
    task_type: detectTaskType(text, docGroup),
    must_include: mustInclude,
    must_not: buildMustNot(docGroup),
    evidence: buildEvidence(docGroup),
    negative_example: buildNegative(text, docGroup),
    notes,
    corpus_note: inUniverse ? null : "70개 universe에 없는 회사 - manifest 확인 또는 질문 회사 교체",
  };
}

function toYaml(d) {
  const lines = [
    `question_id: ${d.question_id}`,
    `question: ${d.question}`,
    `listed_name: ${d.listed_name}`,
    `corp_name: ${d.corp_name}`,
    `doc_group: ${d.doc_group}`,
    `report_nm · period: ${d.report_hint}`,
    `basis: ${d.basis}`,
    `task_type (기대): ${d.task_type}`,
    "must_include_in_answer:",
    ...d.must_include.map((x) => `  - ${x}`),
    `must_NOT_invent: ${d.must_not}`,
    "evidence:",
    ...d.evidence.map((x) => `  ${x}`),
    `negative_example: ${d.negative_example}`,
    `notes: ${d.notes}`,
  ];
  if (d.corpus_note) lines.push(`corpus_note: ${d.corpus_note}`);
  return lines.join("\n");
}

function renderSingle(draft) {
  const section = document.getElementById("single-out");
  if (!draft) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  document.getElementById("pills").innerHTML = [
    `<span class="pill">${draft.doc_group}</span>`,
    `<span class="pill">${draft.task_type.split(" · ")[0]}</span>`,
    draft.basis !== "(질문에 연결/별도 명시 권장)" && draft.basis !== "해당없음"
      ? `<span class="pill warn">${draft.basis}</span>`
      : "",
    draft.corpus_note ? `<span class="pill warn">universe 밖</span>` : "",
  ].join("");
  const rows = [
    ["question_id", draft.question_id],
    ["question", draft.question],
    ["listed_name", draft.listed_name],
    ["corp_name", draft.corp_name],
    ["doc_group", draft.doc_group],
    ["report_nm · period", draft.report_hint],
    ["basis", draft.basis],
    ["task_type", draft.task_type],
    ["must_include", draft.must_include.join(" / ")],
    ["must_NOT", draft.must_not],
    ["negative_example", draft.negative_example],
    ["notes", draft.notes],
  ];
  if (draft.corpus_note) rows.push(["corpus_note", draft.corpus_note]);
  document.querySelector("#field-table tbody").innerHTML = rows
    .map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(v)}</td></tr>`)
    .join("");
  document.getElementById("yaml-out").textContent = toYaml(draft);
}

function renderBatch(drafts) {
  const section = document.getElementById("batch-out");
  if (!drafts.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  document.getElementById("batch-list").innerHTML = drafts
    .map(
      (d, i) =>
        `<div style="margin-bottom:16px"><strong>${escapeHtml(d.question_id)}</strong> ` +
        `<span class="pill">${d.doc_group}</span>` +
        `<pre>${escapeHtml(toYaml(d))}</pre></div>`,
    )
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function refresh() {
  const prefix = document.getElementById("prefix").value.trim();
  const question = document.getElementById("question").value;
  renderSingle(buildDraft(question, prefix, 0));
  const batchLines = document
    .getElementById("batch")
    .value.split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const batchDrafts = batchLines.map((line, i) => buildDraft(line, prefix || `B${i + 1}`, i + 1)).filter(Boolean);
  renderBatch(batchDrafts);
}

async function init() {
  try {
    const res = await fetch("universe.json");
    universe = await res.json();
  } catch {
    universe = [];
  }
  ["question", "batch", "prefix"].forEach((id) => {
    document.getElementById(id).addEventListener("input", refresh);
  });
  refresh();
}

init();
