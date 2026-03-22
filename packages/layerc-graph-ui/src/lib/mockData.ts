import type { LayerCCasePayload } from "./types";

/** Demo episodes for list page */
export const MOCK_EPISODE_INDEX: Array<{
  episode_id: string;
  triage_level: string;
  alert_count: number;
  top_entity: string;
  top_action: string;
  status: string;
}> = [
  {
    episode_id: "ep-demo-wazuh-001",
    triage_level: "suspicious",
    alert_count: 102,
    top_entity: "web-01.internal",
    top_action: "watchlist",
    status: "completed",
  },
  {
    episode_id: "ep-demo-cert-002",
    triage_level: "noise",
    alert_count: 12,
    top_entity: "vpn-gw-03",
    top_action: "collect_more_data",
    status: "completed",
  },
];

const demoCase: LayerCCasePayload = {
  episode_id: "ep-demo-wazuh-001",
  run_id: "run-mock-1",
  status: "completed",
  target_ip: "10.0.0.50",
  alert_count: 102,
  episode: {
    episode_id: "ep-demo-wazuh-001",
    run_id: "run-mock-1",
    t0_ms: Date.now() - 86_400_000,
    t1_ms: Date.now(),
    entities: ["web-01.internal", "10.0.0.50"],
    artifacts: [
      { type: "ip", value: "10.0.0.50" },
      { type: "file", value: "C:\\\\Windows\\\\Temp\\\\a.exe" },
    ],
    sequence_tags: ["lateral"],
    risk_context: { note: "demo" },
  },
  evidence_json: {
    episode_id: "ep-demo-wazuh-001",
    run_id: "run-mock-1",
    items: [
      {
        evidence_id: "ev-sop-1",
        source: "kb",
        kind: "snippet",
        title: "sop_incident_response.md",
        body: "Escalate when lateral movement is suspected…",
        score: 0.82,
      },
      {
        evidence_id: "ev-sop-2",
        source: "kb",
        kind: "snippet",
        title: "sop_insider_anomaly.md",
        body: "Review user sessions for anomalous access…",
        score: 0.71,
      },
      {
        evidence_id: "ev-hunt-1",
        source: "kb",
        kind: "snippet",
        title: "hunt_query_templates.md",
        body: "Wazuh: agent.name + rule.groups…",
        score: 0.65,
      },
      {
        evidence_id: "ev-policy-1",
        source: "kb",
        kind: "snippet",
        title: "response_policy_guardrails.md",
        body: "No isolate without critical triage…",
        score: 0.9,
      },
      {
        evidence_id: "ev-cti-1",
        source: "opencti",
        kind: "stix_object",
        title: "Indicator: malicious hash",
        body: "Associated with commodity loader…",
        score: 0.55,
      },
    ],
    stats: { count: 5 },
  },
  agent_outputs_json: {
    by_rule: {
      default: {
        triage: {
          agent_id: "triage",
          episode_id: "ep-demo-wazuh-001",
          run_id: "run-mock-1",
          summary:
            "Triage: suspicious. Lateral movement tags present. Key risks: host pivot; data staging.",
          confidence: 0.72,
          citations: ["ev-sop-1", "ev-sop-2", "ev-cti-1"],
          assumptions: ["Alerts are complete for the time window."],
          next_required_data: ["EDR process tree for web-01"],
          structured: {
            triage_level: "suspicious",
            why_now: "Spike in alerts + lateral tag.",
            top_evidence: ["ev-sop-1"],
            key_risks: ["lateral movement", "credential reuse"],
          },
        },
        hunt_planner: {
          agent_id: "hunt_planner",
          episode_id: "ep-demo-wazuh-001",
          run_id: "run-mock-1",
          summary: "Prioritize auth + network pivots on web-01 and 10.0.0.50.",
          confidence: 0.68,
          citations: ["ev-hunt-1", "ev-sop-1"],
          assumptions: ["Index retention covers 24h window."],
          next_required_data: ["Proxy logs if available"],
          structured: {
            queries: [
              { wazuh_query: "agent.name:web-01 AND rule.level:>=10" },
            ],
            pivots: ["same-user different-host"],
            expected_findings: ["repeated failed auth"],
          },
        },
        response_advisor: {
          agent_id: "response_advisor",
          episode_id: "ep-demo-wazuh-001",
          run_id: "run-mock-1",
          summary:
            "Response: Watchlist; isolate if supported. Possible isolation if exfil confirmed.",
          confidence: 0.7,
          citations: ["ev-policy-1", "ev-sop-1", "ev-hunt-1"],
          assumptions: [
            "Evidence and triage/hunt outputs are the only basis for actions.",
          ],
          next_required_data: [],
          structured: {
            actions: [
              {
                action: "collect_more_data",
                target: "web-01.internal,10.0.0.50",
                duration_minutes: null,
                guardrails:
                  "Do not block or isolate until triage confirms; scope collection.",
                rollback_conditions: ["If triage downgrades to noise, cancel collection."],
              },
              {
                action: "watchlist",
                target: "web-01.internal,10.0.0.50",
                duration_minutes: 120,
                guardrails: "Monitor only; escalate to isolate if lateral confirmed.",
                rollback_conditions: [
                  "Remove from watchlist when hypothesis is refuted or contained.",
                ],
              },
            ],
            expected_impact:
              "Minimal impact (watchlist/collect_more_data) unless isolation triggers.",
            scope: ["web-01.internal", "10.0.0.50"],
            rollback_conditions: ["Revert on false positive."],
            duration_minutes: 120,
          },
        },
      },
    },
    triage: undefined,
    hunt_planner: undefined,
    response_advisor: undefined,
  },
  writeback_json: {
    notes: [
      {
        kind: "decision",
        text: "Dry-run writeback recorded; no production change.",
        derived_from: "response_advisor",
      },
    ],
    mode: "dry_run",
  },
};

/** Primary mock graph payload */
export function getMockCaseByEpisodeId(episodeId: string): LayerCCasePayload | null {
  if (episodeId === "ep-demo-wazuh-001") {
    return { ...demoCase, episode_id: episodeId };
  }
  if (episodeId === "ep-demo-cert-002") {
    return {
      ...demoCase,
      episode_id: "ep-demo-cert-002",
      alert_count: 12,
      episode: {
        ...demoCase.episode!,
        episode_id: "ep-demo-cert-002",
        entities: ["vpn-gw-03"],
        artifacts: [{ type: "ip", value: "203.0.113.5" }],
        sequence_tags: [],
      },
      evidence_json: {
        ...demoCase.evidence_json!,
        episode_id: "ep-demo-cert-002",
        items: demoCase.evidence_json!.items.slice(0, 3),
      },
    };
  }
  return null;
}
