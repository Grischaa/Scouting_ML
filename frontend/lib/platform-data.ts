import {
  activityFeed,
  players,
  reportBlocks as baseReportBlocks,
  shortlists as baseShortlists,
  teamCards as baseTeamCards,
  topPerformersByRole,
  watchlistTasks,
} from "@/lib/mock-data";
import type {
  ConfidenceLevel,
  DecisionStatus,
  FilterPreset,
  KPI,
  NotificationItem,
  Player,
  PlayerIntel,
  PlayerPosition,
  PlayerProfile,
  ReportBlock,
  TeamCard,
  TeamNeed,
  Workspace,
} from "@/lib/types";

const playerIntelSeed: Record<
  string,
  Omit<PlayerIntel, "playerId" | "valueGapM" | "valueGapPct" | "confidenceLevel">
> = {
  p1: {
    predictedValueM: 11.8,
    confidenceScore: 88,
    decisionStatus: "Pursue",
    nextAction: "Add to sporting director review",
    decisionReason: "Role scarcity, repeatable carries, and a controllable fee make him a board-level fullback case.",
    reliabilityNote: "Stable Eredivisie minutes and repeatable final-third output across two tactical contexts.",
    roleFitLabel: "Aggressive width fullback",
    roleFitScore: 91,
    formationFits: ["4-3-3", "3-4-3", "4-2-3-1"],
    contractUrgency: "Controlled until June 2027",
    readiness: "Ready now",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Moderate",
    ageCurve: "Pre-peak fullback with immediate resale runway.",
    squadNeed: "Left-back progression with high running volume",
    availability: "Summer 2026 realistic",
    reportFocus: "Live check back-post defending and crossing decisions under pressure.",
    modelFlags: ["High value gap", "Non-Big-5 market", "Repeat sprint volume"],
    watchlistOwner: "L. De Smet",
    shortlistCount: 3,
    consultantAngle: "Strong consultant case for clubs needing a starting fullback below top-tier pricing.",
  },
  p2: {
    predictedValueM: 12.4,
    confidenceScore: 91,
    decisionStatus: "Pursue",
    nextAction: "Assign live scouting",
    decisionReason: "Contract leverage and a stable midfield floor make him the clearest live action in the market.",
    reliabilityNote: "Strong minute load, high-possession role security, and low-volatility ball-winning profile.",
    roleFitLabel: "Anchor six for 4-3-3 / 4-2-3-1",
    roleFitScore: 94,
    formationFits: ["4-3-3", "4-2-3-1", "3-2-4-1"],
    contractUrgency: "One summer from leverage point",
    readiness: "Ready now",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Low",
    ageCurve: "Prime development phase for a possession anchor.",
    squadNeed: "Midfield control and rest-defence structure",
    availability: "Accessible before contract pressure hardens",
    reportFocus: "Validate range against transition-heavy opponents before board sign-off.",
    modelFlags: ["Contract opportunity", "Press-resistant pivot", "Immediate floor"],
    watchlistOwner: "N. Rossi",
    shortlistCount: 4,
    consultantAngle: "Easy board narrative: lowers chaos and gives staff a ready-now midfield stabiliser.",
  },
  p3: {
    predictedValueM: 9.6,
    confidenceScore: 79,
    decisionStatus: "Review",
    nextAction: "Assign live scouting",
    decisionReason: "The upside is strong, but live evidence is still needed on final action and defensive load.",
    reliabilityNote: "Chance creation is strong, but defensive load and end-product still swing game to game.",
    roleFitLabel: "Direct winger for front-foot 4-3-3",
    roleFitScore: 87,
    formationFits: ["4-3-3", "4-2-3-1", "3-4-2-1"],
    contractUrgency: "Monitor before June 2026 leverage opens",
    readiness: "Ready now",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Moderate",
    ageCurve: "Near-prime creator with output upside if team context is aggressive.",
    squadNeed: "Wide isolation and chance creation",
    availability: "Summer 2026 realistic",
    reportFocus: "Need live view on defensive concentration and action selection late in possessions.",
    modelFlags: ["Wide creation", "Belgium market", "High-value dribbler"],
    watchlistOwner: "A. Mertens",
    shortlistCount: 2,
    consultantAngle: "Useful for clubs that need a visible difference-maker without paying top-five winger money.",
  },
  p4: {
    predictedValueM: 8.1,
    confidenceScore: 74,
    decisionStatus: "Watch",
    nextAction: "Keep on watchlist",
    decisionReason: "Creative value is attractive, but physical translation risk keeps him in the watch lane.",
    reliabilityNote: "Creative indicators are strong, but physical profile limits some league translations.",
    roleFitLabel: "Narrow creator between lines",
    roleFitScore: 83,
    formationFits: ["4-2-3-1", "4-3-3", "4-4-2 diamond"],
    contractUrgency: "Approaching 2026 decision point",
    readiness: "Rotation-ready",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Moderate",
    ageCurve: "Technical peak runway remains positive if protected physically.",
    squadNeed: "Final-third connector and set-passing quality",
    availability: "Value likely rises with another stable season",
    reportFocus: "Pressure test against physical, transition-heavy league profile.",
    modelFlags: ["Creative upside", "Possession specialist", "Fair wage profile"],
    watchlistOwner: "S. Ainsworth",
    shortlistCount: 2,
    consultantAngle: "Ideal for consultants building stylistic alternatives to expensive ten-profile markets.",
  },
  p5: {
    predictedValueM: 10.2,
    confidenceScore: 85,
    decisionStatus: "Review",
    nextAction: "Add to sporting director review",
    decisionReason: "Left-footed centre-back scarcity and a strong floor justify a sporting director review.",
    reliabilityNote: "Strong defending floor, left-sided scarcity, and stable Belgian development pathway.",
    roleFitLabel: "Front-foot left centre-back",
    roleFitScore: 89,
    formationFits: ["4-3-3", "3-4-3", "4-2-3-1"],
    contractUrgency: "Secure contract, but valuation still below profile tier",
    readiness: "Ready now",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Moderate",
    ageCurve: "Defender entering the strongest asset-building years.",
    squadNeed: "Left-footed progression from first line",
    availability: "Possible if club sells as development step",
    reportFocus: "Open-field speed ceiling and exposure in very high line.",
    modelFlags: ["Left-sided scarcity", "Belgian defender", "Stable progression data"],
    watchlistOwner: "E. Vandenberg",
    shortlistCount: 3,
    consultantAngle: "A sellable profile for clubs chasing left-footed centre-back scarcity without Premier League pricing.",
  },
  p6: {
    predictedValueM: 7.3,
    confidenceScore: 77,
    decisionStatus: "Price Check",
    nextAction: "Use for price reference",
    decisionReason: "He is a dependable floor option, but only at a disciplined fee.",
    reliabilityNote: "Defensive actions are repeatable, but build-up ceiling suppresses model certainty.",
    roleFitLabel: "Defensive stopper for hybrid back lines",
    roleFitScore: 78,
    formationFits: ["4-4-2", "3-4-3", "4-2-3-1"],
    contractUrgency: "Potential leverage as 2026 approaches",
    readiness: "Ready now",
    priceRealism: "Fair value",
    injuryRisk: "Low",
    adaptationRisk: "Low",
    ageCurve: "Useful floor profile with modest upside left.",
    squadNeed: "Aerial security and recovery defending",
    availability: "Good value if price remains disciplined",
    reportFocus: "Need stronger evidence of passing range under pressure.",
    modelFlags: ["Aerial dominance", "Defensive floor", "Lower build-up ceiling"],
    watchlistOwner: "P. Moller",
    shortlistCount: 1,
    consultantAngle: "Works as a pragmatic recommendation when floor matters more than resale story.",
  },
  p7: {
    predictedValueM: 8.8,
    confidenceScore: 84,
    decisionStatus: "Review",
    nextAction: "Add to sporting director review",
    decisionReason: "Contract timing and control in possession make him a live review candidate.",
    reliabilityNote: "High technical repeatability and contract context improve trust despite moderate athletic concerns.",
    roleFitLabel: "Control midfielder in double pivot",
    roleFitScore: 90,
    formationFits: ["4-2-3-1", "4-3-3", "3-2-4-1"],
    contractUrgency: "Leverage window opening before June 2026",
    readiness: "Rotation-ready",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Moderate",
    ageCurve: "Skill-led midfielder entering valuable contract years.",
    squadNeed: "Press-resistant connector and tempo control",
    availability: "Realistic this window",
    reportFocus: "Assess how much athletic support he needs around him.",
    modelFlags: ["Contract watch", "Possession fit", "High ball security"],
    watchlistOwner: "R. Delaunay",
    shortlistCount: 3,
    consultantAngle: "Strong fit for consultants pitching a lower-cost control midfielder to multi-club groups.",
  },
  p8: {
    predictedValueM: 10.8,
    confidenceScore: 81,
    decisionStatus: "Watch",
    nextAction: "Keep on watchlist",
    decisionReason: "The striker upside is real, but translation risk is still too high for a push now.",
    reliabilityNote: "The upside is clear, but striker translation still carries more context risk than midfield profiles.",
    roleFitLabel: "Space-attacking striker",
    roleFitScore: 88,
    formationFits: ["4-4-2", "4-2-3-1", "3-4-1-2"],
    contractUrgency: "Long deal reduces leverage; timing matters",
    readiness: "Developmental",
    priceRealism: "Undervalued",
    injuryRisk: "Low",
    adaptationRisk: "Elevated",
    ageCurve: "High-upside striker before physical and tactical peak.",
    squadNeed: "Depth-running nine with resale potential",
    availability: "Possible only before another scoring spike",
    reportFocus: "Live view on link play and out-of-possession intensity.",
    modelFlags: ["U21 striker", "Transition threat", "Turkish market"],
    watchlistOwner: "B. Kaya",
    shortlistCount: 2,
    consultantAngle: "Useful for consultant decks where upside and future sale are central to the investment case.",
  },
  p9: {
    predictedValueM: 5.1,
    confidenceScore: 68,
    decisionStatus: "Pass",
    nextAction: "Drop from active pursuit",
    decisionReason: "Goalkeeper variance remains too high to keep him in the active recruitment lane.",
    reliabilityNote: "Keeper output is more volatile, and the shot-stopping floor remains less settled.",
    roleFitLabel: "Distribution-first goalkeeper",
    roleFitScore: 76,
    formationFits: ["4-3-3", "4-2-3-1", "3-4-3"],
    contractUrgency: "Decision point approaching in 2026",
    readiness: "Rotation-ready",
    priceRealism: "Fair value",
    injuryRisk: "Low",
    adaptationRisk: "Moderate",
    ageCurve: "Goalkeeper still early in senior development curve.",
    squadNeed: "Build-up support from goalkeeper line",
    availability: "Available if shot-stopping concerns keep market cool",
    reportFocus: "Need more live evidence on near-post handling and low-shot concentration.",
    modelFlags: ["Goalkeeper market", "Distribution value", "Higher variance"],
    watchlistOwner: "F. Keller",
    shortlistCount: 1,
    consultantAngle: "Only compelling for clubs prioritising build-up style over pure shot-stop certainty.",
  },
  p10: {
    predictedValueM: 8.7,
    confidenceScore: 72,
    decisionStatus: "Price Check",
    nextAction: "Use for price reference",
    decisionReason: "The profile is useful, but the current market already prices in too much of the upside.",
    reliabilityNote: "Action volume is real, but current market price already reflects much of the profile quality.",
    roleFitLabel: "High-motor number eight",
    roleFitScore: 80,
    formationFits: ["4-3-3", "4-4-2 diamond", "4-2-3-1"],
    contractUrgency: "No immediate leverage",
    readiness: "Ready now",
    priceRealism: "Fair value",
    injuryRisk: "Low",
    adaptationRisk: "Low",
    ageCurve: "Close to fair market value at current development point.",
    squadNeed: "Energy and transition coverage from midfield",
    availability: "Only viable if budget stretches",
    reportFocus: "Need to prove enough ball security to justify current fee.",
    modelFlags: ["Fair value", "High motor", "Limited discount"],
    watchlistOwner: "G. Vos",
    shortlistCount: 1,
    consultantAngle: "Use as a benchmark more than a true undervaluation story.",
  },
  p11: {
    predictedValueM: 6.5,
    confidenceScore: 71,
    decisionStatus: "Watch",
    nextAction: "Keep on watchlist",
    decisionReason: "The pace and box threat are interesting, but the evidence base is still thin.",
    reliabilityNote: "Explosive output flashes are attractive, but overall game still carries a large refinement gap.",
    roleFitLabel: "Vertical inside forward",
    roleFitScore: 79,
    formationFits: ["4-3-3", "4-2-3-1", "3-4-2-1"],
    contractUrgency: "Secure contract but still affordable",
    readiness: "Developmental",
    priceRealism: "Undervalued",
    injuryRisk: "Moderate",
    adaptationRisk: "Elevated",
    ageCurve: "Upside asset if decision-making improves quickly.",
    squadNeed: "High-variance pace and box running off the wing",
    availability: "Accessible before broader market catches up",
    reportFocus: "Need more evidence on coachability and defensive discipline.",
    modelFlags: ["High variance", "Wide speed", "Primeira Liga upside"],
    watchlistOwner: "Y. Mbaye",
    shortlistCount: 1,
    consultantAngle: "Useful upside recommendation for clubs happy to absorb developmental volatility.",
  },
  p12: {
    predictedValueM: 5.8,
    confidenceScore: 75,
    decisionStatus: "Price Check",
    nextAction: "Use for price reference",
    decisionReason: "Reliable depth option, best used to anchor price discipline rather than active pursuit.",
    reliabilityNote: "Durability and defensive floor boost trust, though upside ceiling is lower than top-value peers.",
    roleFitLabel: "Balanced defensive fullback",
    roleFitScore: 77,
    formationFits: ["4-4-2", "4-2-3-1", "3-5-2"],
    contractUrgency: "2026 leverage could create discount",
    readiness: "Ready now",
    priceRealism: "Fair value",
    injuryRisk: "Low",
    adaptationRisk: "Low",
    ageCurve: "Closer to established level than breakout profile.",
    squadNeed: "Defensive left-back depth",
    availability: "Good fallback if priority board inflates",
    reportFocus: "Benchmark against more progressive fullback options before moving.",
    modelFlags: ["Reliable floor", "Greek market", "Lower upside"],
    watchlistOwner: "I. Petrou",
    shortlistCount: 1,
    consultantAngle: "Clean fallback option for clubs prioritising reliability over upside.",
  },
};

export const decisionPriorityMap: Record<DecisionStatus, number> = {
  Pursue: 5,
  Review: 4,
  Watch: 3,
  "Price Check": 2,
  Pass: 1,
};

export const confidenceLevelMap = (score: number): ConfidenceLevel => {
  if (score >= 84) return "High confidence";
  if (score >= 74) return "Caution";
  return "Thin evidence";
};

export const confidencePriorityMap: Record<ConfidenceLevel, number> = {
  "High confidence": 3,
  Caution: 2,
  "Thin evidence": 1,
};

export function sortProfilesByDecision(left: PlayerProfile, right: PlayerProfile) {
  const decisionGap = decisionPriorityMap[right.intel.decisionStatus] - decisionPriorityMap[left.intel.decisionStatus];
  if (decisionGap !== 0) return decisionGap;

  const valueGap = right.intel.valueGapM - left.intel.valueGapM;
  if (valueGap !== 0) return valueGap;

  const confidenceGap = right.intel.confidenceScore - left.intel.confidenceScore;
  if (confidenceGap !== 0) return confidenceGap;

  return right.player.scoutingScore - left.player.scoutingScore;
}

export function getPlayerIntel(player: Player): PlayerIntel {
  const seed = playerIntelSeed[player.id];
  const valueGapM = Number((seed.predictedValueM - player.marketValueM).toFixed(1));
  const valueGapPct = Math.round((valueGapM / player.marketValueM) * 100);

  return {
    ...seed,
    playerId: player.id,
    valueGapM,
    valueGapPct,
    confidenceLevel: confidenceLevelMap(seed.confidenceScore),
  };
}

export const playerProfiles: PlayerProfile[] = players.map((player) => ({
  player,
  intel: getPlayerIntel(player),
}));

export const playerProfileMap = Object.fromEntries(
  playerProfiles.map((profile) => [profile.player.id, profile]),
) as Record<string, PlayerProfile>;

export const workspaces: Workspace[] = [
  { id: "w1", name: "First Team Recruitment", scope: "Summer 2026 window", status: "Board review on Monday" },
  { id: "w2", name: "Consultancy Clients", scope: "6 external mandates", status: "2 decks due this week" },
  { id: "w3", name: "U23 Succession", scope: "Non-Big-5 pipeline", status: "18 live profiles" },
];

export const notifications: NotificationItem[] = [
  { id: "n1", label: "Value ceiling drift", detail: "Joao Serrano's market signal tightened by 4% this week.", tone: "amber", time: "15m" },
  { id: "n2", label: "Live report added", detail: "New winger memo attached to Rafik Belghali.", tone: "blue", time: "1h" },
  { id: "n3", label: "Contract trigger", detail: "Martim Neto enters the final controllable year on 2026-06-30.", tone: "green", time: "Today" },
];

export const commandSearchHints = [
  "Search player, league, archetype, or report",
  "Try 'U23 left-backs outside Big 5'",
  "Try 'contract opportunities expiring 2026'",
];

const countProfiles = (predicate: (profile: PlayerProfile) => boolean) => playerProfiles.filter(predicate).length;

export const executiveKpis: KPI[] = [
  {
    label: "Pursue now",
    value: String(countProfiles((profile) => profile.intel.decisionStatus === "Pursue")),
    delta: `${countProfiles((profile) => profile.intel.nextAction === "Assign live scouting")} live scouting call(s)`,
    tone: "green",
    context: "Top-priority actions with strong value and conviction",
  },
  {
    label: "Review this week",
    value: String(countProfiles((profile) => profile.intel.decisionStatus === "Review")),
    delta: `${countProfiles((profile) => profile.intel.nextAction === "Add to sporting director review")} board review case(s)`,
    tone: "blue",
    context: "Cases credible enough for internal decision meetings",
  },
  {
    label: "Watchlist holds",
    value: String(countProfiles((profile) => profile.intel.decisionStatus === "Watch")),
    delta: "Keep the evidence base warm",
    tone: "amber",
    context: "Interesting names that still need patience",
  },
  {
    label: "Price anchors",
    value: String(countProfiles((profile) => profile.intel.decisionStatus === "Price Check")),
    delta: `${countProfiles((profile) => profile.intel.nextAction === "Use for price reference")} benchmark case(s)`,
    tone: "blue",
    context: "Useful for negotiating discipline, not active chase",
  },
  {
    label: "High confidence",
    value: String(countProfiles((profile) => profile.intel.confidenceLevel === "High confidence")),
    delta: "Stable evidence behind the call",
    tone: "green",
    context: "Most reliable opportunity subset in the current board",
  },
  {
    label: "Parked cases",
    value: String(countProfiles((profile) => profile.intel.decisionStatus === "Pass")),
    delta: "Kept visible for audit only",
    tone: "amber",
    context: "Out of the active lane until evidence changes",
  },
];

export const opportunityBoard = [...playerProfiles]
  .sort(sortProfilesByDecision)
  .slice(0, 6);

export const trendingProfiles = playerProfiles
  .filter(({ player }) => player.form !== "Cooling")
  .sort(sortProfilesByDecision)
  .slice(0, 5);

export const roleLeaders = topPerformersByRole.map((player) => ({
  player,
  intel: getPlayerIntel(player),
}));

export const confidenceSummary = [
  {
    label: "High confidence",
    value: playerProfiles.filter((profile) => profile.intel.confidenceLevel === "High confidence").length,
    note: "Evidence is strong enough to move without another interpretation loop.",
  },
  {
    label: "Caution",
    value: playerProfiles.filter((profile) => profile.intel.confidenceLevel === "Caution").length,
    note: "The case is real, but action should stay conditional on one more check.",
  },
  {
    label: "Thin evidence",
    value: playerProfiles.filter((profile) => profile.intel.confidenceLevel === "Thin evidence").length,
    note: "Upside may exist, but the proof is still too fragile for a strong push.",
  },
];

export const valueGapConfidenceData = playerProfiles.map(({ player, intel }) => ({
  name: player.name,
  position: player.position,
  league: player.league,
  valueGap: intel.valueGapM,
  confidence: intel.confidenceScore,
  scoutingScore: player.scoutingScore,
}));

export const dashboardActivity = activityFeed;
export const taskBoard = watchlistTasks;

export const discoveryFilterPresets: FilterPreset[] = [
  { id: "f1", label: "Pursue-ready value", description: "Profiles with real gap, strong evidence, and a live next step." },
  { id: "f2", label: "Board review cases", description: "Players close enough to move into sporting director discussion." },
  { id: "f3", label: "Watchlist with upside", description: "Good names to hold without forcing them into the active lane." },
];

export const discoveryOptions = {
  positions: ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"] as PlayerPosition[],
  leagues: [...new Set(players.map((player) => player.league))].sort(),
  feet: ["Right", "Left", "Both"] as const,
  archetypes: [...new Set(players.map((player) => player.archetype))].sort(),
  readiness: ["Ready now", "Rotation-ready", "Developmental"] as const,
  confidence: ["High confidence", "Caution", "Thin evidence"] as const,
};

export const trackedProfiles = [...playerProfiles]
  .sort(sortProfilesByDecision)
  .slice(0, 9);

export const compareDefaultIds = ["p1", "p2", "p8"];

export const reportTemplates: ReportBlock[] = [
  ...baseReportBlocks,
  { id: "fit", label: "Recruitment Fit", description: "Role, formation, squad-need, and game-model alignment block." },
  { id: "value", label: "Contract / Value Summary", description: "Predicted value, market value, and leverage position for negotiation." },
  { id: "consultant", label: "Decision Summary", description: "Client-ready framing for the call, the conviction, and the next step." },
];

export const shortlistProfiles = baseShortlists.map((shortlist) => ({
  shortlist,
  profiles: shortlist.playerIds.map((id) => playerProfileMap[id]).filter(Boolean),
}));

export const reportFocusPlayer = playerProfileMap.p2;

export const teamNeeds: TeamNeed[] = [
  {
    id: "t1",
    team: "Royal Antwerp",
    league: "Belgian Pro League",
    formation: "4-2-3-1",
    need: "Progressive left-back",
    fitSummary: "Need width and repeat carries without losing defensive balance in rest-defence.",
    valueFocus: "Buy before the market shifts into €10m+ territory.",
  },
  {
    id: "t2",
    team: "SC Braga",
    league: "Primeira Liga",
    formation: "4-4-2 diamond",
    need: "Control midfielder",
    fitSummary: "Need a deep connector who can stabilise first phase and protect central space.",
    valueFocus: "Target contract-leveraged players with immediate floor.",
  },
  {
    id: "t3",
    team: "FC Twente",
    league: "Eredivisie",
    formation: "4-3-3",
    need: "Direct wide threat",
    fitSummary: "Need isolation winger who can create separation and supply final-third actions quickly.",
    valueFocus: "Non-Big-5 creator with consultant-ready resale story.",
  },
];

export const teamContextCards: TeamCard[] = baseTeamCards;

export function getProfilesByIds(ids: string[]) {
  return ids.map((id) => playerProfileMap[id]).filter(Boolean);
}

export function getRoleLeaders(position: PlayerPosition) {
  return playerProfiles
    .filter((profile) => profile.player.position === position)
    .sort((left, right) => right.player.scoutingScore - left.player.scoutingScore);
}
