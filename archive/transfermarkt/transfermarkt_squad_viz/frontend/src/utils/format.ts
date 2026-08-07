export function formatMarketValue(eur: number | null | undefined): string {
  if (eur == null || eur <= 0) return "—";
  if (eur >= 1_000_000) return `€${(eur / 1_000_000).toFixed(2)}m`;
  if (eur >= 1_000) return `€${(eur / 1_000).toFixed(0)}k`;
  return `€${eur}`;
}

export function formatDateRange(from: string, to: string | null): string {
  if (!to) return `${from} → ongoing`;
  return `${from} → ${to}`;
}
