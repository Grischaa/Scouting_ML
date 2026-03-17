export type PlayerPosition = "GK" | "CB" | "FB" | "DM" | "CM" | "AM" | "W" | "ST";
export type ScoutStatus = "monitor" | "shortlist" | "priority" | "watch";
export type Recommendation = "Monitor" | "Shortlist" | "Sign" | "Not Suitable";

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
    recommendation: Recommendation;
  };
  tags: string[];
  radar: Array<{ subject: string; value: number }>;
  percentiles: PercentileMetric[];
  performanceSplit: Array<{ category: string; current: number; benchmark: number }>;
  trend: TrendPoint[];
  marketTrend: TrendPoint[];
  matchLogs: MatchLog[];
  similarityIds: string[];
  scoutNotes: Array<{ author: string; note: string; time: string }>;
}

export interface KPI {
  label: string;
  value: string;
  delta: string;
  tone: "green" | "blue" | "amber";
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
