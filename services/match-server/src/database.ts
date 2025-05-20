import Database from "better-sqlite3"

export const Pong_Hist_db = new Database('pong_hist.db')

Pong_Hist_db.exec(`
    CREATE TABLE IF NOT EXISTS MatchResult (
        game INTEGER DEFAULT 0,
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        playerA_id INTEGER NOT NULL,
        playerB_id INTEGER NOT NULL,
        scoreA INTEGER NOT NULL,
        scoreB INTEGER NOT NULL,
        winner_id INTEGER NOT NULL,
        match_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
`)

export type MatchResult = {
    game: number;
    id: number;
    playerA_id: number;
    playerB_id: number;
    scoreA: number;
    scoreB: number;
    winner_id: number;
    match_time: string; // format ISO
};

export function insertMatchResult(
    playerA_id: number,
    playerB_id: number,
    scoreA: number,
    scoreB: number,
    winner: number
) {
    let winner_id = -3; // -3 means draw game
    if (winner === 0)
        winner_id = playerA_id;
    else if (winner === 1)
        winner_id = playerB_id;
    console.log(`winner id: ${winner_id}`);
    Pong_Hist_db.prepare(`
        INSERT INTO MatchResult (playerA_id, playerB_id, scoreA, scoreB, winner_id)
        VALUES (?, ?, ?, ?, ?)
    `).run(playerA_id, playerB_id, scoreA, scoreB, winner_id);
}

export function getMatchHistory(userId: number): MatchResult[] {
    return Pong_Hist_db.prepare(`
        SELECT * FROM MatchResult
        WHERE playerA_id = ? OR playerB_id = ?
        ORDER BY match_time DESC
    `).all(userId, userId) as MatchResult[];
}

export function getWinnerId(playerA_id: number, playerB_id: number): number | null {
    const matches = getMatchHistory(playerA_id).filter(
        (match) =>
            (match.playerA_id === playerA_id && match.playerB_id === playerB_id) ||
            (match.playerA_id === playerB_id && match.playerB_id === playerA_id)
    );
    if (matches.length === 0)
        return null;
    const lastMatch = matches[0];
    return (lastMatch.winner_id);
}