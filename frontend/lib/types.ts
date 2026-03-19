export type PlayerPosition = "GK" | "CB" | "FB" | "DM" | "CM" | "AM" | "W" | "ST";
export type ScoutStatus = "monitor" | "shortlist" | "priority" | "watch";
export type DecisionStatus = "Pursue" | "Review" | "Watch" | "Price Check" | "Pass";
export type NextAction =
  | "Assign live scouting"
  | "Add to sporting director review"
  | "Keep on watchlist"
  | "Use for price reference"
  | "Drop from active pursuit";
export type ConfidenceLevel = "High confidence" | "Caution" | "Thin evidence";
export type ReadinessBand = "Ready now" | "Rotation-ready" | "Developmental";
export type RiskBand = "Low" | "Moderate" | "Elevated";
export type Tone = "neutral" | "green" | "blue" | "amber" | "red";

export interface TrendPoint {
  label: string;
  value: number;
}

export interface MatchLog {
  opponent: string;
  date: string;
  rating: number;
  minutes: number;
  goals: number;
  assists: number;
  progressivePasses: number;
  duelsWon: number;
}

export interface PercentileMetric {
  label: string;
  value: number;
  tone?: "green" | "blue" | "amber";
}

export interface ScoutNote {
  author: string;
  note: string;
  time: string;
  type?: "Live" | "Video" | "Model";
}

export interface Player {
  id: string;
  slug: string;
  name: string;
  age: number;
  nationality: string;
  club: string;
  league: string;
  position: PlayerPosition;
  secondaryPositions: PlayerPosition[];
  minutes: number;
  marketValueM: number;
  scoutingScore: number;
  form: "Rising" | "Stable" | "Cooling";
  contractExpiry: string;
  preferredFoot: "Left" | "Right" | "Both";
  heightCm: number;
  status: ScoutStatus;
  archetype: string;
  summary: string;
  strengths: string[];
  concerns: string[];
  report: {
    technical: string;
    tactical: string;
    physical: string;
    mental: string;
    recommendation: DecisionStatus;
  };
  tags: string[];
  radar: Array<{ subject: string; value: number }>;
  percentiles: PercentileMetric[];
  performanceSplit: Array<{ category: string; current: number; benchmark: number }>;
  trend: TrendPoint[];
  marketTrend: TrendPoint[];
  matchLogs: MatchLog[];
  similarityIds: string[];
  scoutNotes: ScoutNote[];
}

export interface PlayerIntel {
  playerId: string;
  predictedValueM: number;
  valueGapM: number;
  valueGapPct: number;
  confidenceScore: number;
  confidenceLevel: ConfidenceLevel;
  decisionStatus: DecisionStatus;
  nextAction: NextAction;
  decisionReason: string;
  reliabilityNote: string;
  roleFitLabel: string;
  roleFitScore: number;
  formationFits: string[];
  contractUrgency: string;
  readiness: ReadinessBand;
  priceRealism: "Undervalued" | "Fair value" | "Stretch fee";
  injuryRisk: RiskBand;
  adaptationRisk: RiskBand;
  ageCurve: string;
  squadNeed: string;
  availability: string;
  reportFocus: string;
  modelFlags: string[];
  watchlistOwner: string;
  shortlistCount: number;
  consultantAngle: string;
}

export interface PlayerProfile {
  player: Player;
  intel: PlayerIntel;
}

export interface KPI {
  label: string;
  value: string;
  delta: string;
  tone: Exclude<Tone, "neutral">;
  context?: string;
}

export interface ActivityItem {
  title: string;
  subtitle: string;
  time: string;
  type: "note" | "report" | "assignment";
}

export interface TaskItem {
  title: string;
  owner: string;
  due: string;
  priority: "High" | "Medium" | "Low";
}

export interface Shortlist {
  id: string;
  name: string;
  count: number;
  averageAge: number;
  averageValueM: number;
  updatedAt: string;
  priority: "Critical" | "Live" | "Exploratory";
  owner: string;
  tags: string[];
  playerIds: string[];
  note: string;
}

export interface ReportBlock {
  id: string;
  label: string;
  description: string;
}

export interface TeamCard {
  name: string;
  league: string;
  style: string;
  pressing: number;
  possession: number;
  averageAge: number;
}

export interface Workspace {
  id: string;
  name: string;
  scope: string;
  status: string;
}

export interface NotificationItem {
  id: string;
  label: string;
  detail: string;
  tone: Exclude<Tone, "neutral">;
  time: string;
}

export interface FilterPreset {
  id: string;
  label: string;
  description: string;
}

export interface TeamNeed {
  id: string;
  team: string;
  league: string;
  formation: string;
  need: string;
  fitSummary: string;
  valueFocus: string;
}
