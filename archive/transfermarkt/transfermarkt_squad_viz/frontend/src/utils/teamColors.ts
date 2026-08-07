/** Known Super League club colors; unknown clubs get a stable HSL fallback. */
const TEAM_COLORS: Record<string, string> = {
  "Olympiacos Piraeus": "#e30613",
  "PAOK Thessaloniki": "#1a1a1a",
  Panathinaikos: "#007a3d",
  "AEK Athens": "#f4c430",
  "Aris Thessaloniki": "#ffd700",
  "OFI Crete": "#006633",
  Panetolikos: "#0057a8",
  Levadiakos: "#0066cc",
  "AE Kifisia": "#003366",
  "Atromitos Athens": "#003399",
  "Asteras Aktor": "#ff6600",
  "AE Larisa": "#8b0000",
  Panserraikos: "#228b22",
  "Volos NFC": "#dc143c",
};

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

export function getTeamColor(club: string): string {
  if (TEAM_COLORS[club]) return TEAM_COLORS[club];
  const hue = hashString(club) % 360;
  return `hsl(${hue}, 55%, 52%)`;
}

export function getTeamColors(): Record<string, string> {
  return { ...TEAM_COLORS };
}
