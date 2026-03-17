import { notFound } from "next/navigation";
import { PlayerProfileView } from "@/components/players/player-profile-view";
import { getPlayerBySlug, getSimilarPlayers } from "@/lib/mock-data";

export default async function PlayerDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const player = getPlayerBySlug(id);

  if (!player) {
    notFound();
  }

  const similarPlayers = getSimilarPlayers(player);

  return <PlayerProfileView player={player} similarPlayers={similarPlayers} />;
}
