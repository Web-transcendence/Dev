import {displayNotification} from "./notificationHandler.js";
import {fetchUserInformation, UserData} from "./user.js";
import {navigate} from "./front.js";

interface ChartData {
    wins: number;
    losses: number;
    mmr: number;
    canvasId: string;
    mmrElementId: string;
}

export type MatchResult = {
    id: number;
    playerA_id: number;
    playerB_id: number;
    scoreA: number;
    scoreB: number;
    winner_id: number;
    match_time: string;
};

function addMatchEntry(game: string, opponent: string, opponentAvatar: string, score1: number, score2: number, result: string, matchTime: string) {
    const template = document.getElementById('matchTemplate') as HTMLTemplateElement;
    const list = document.getElementById('matchHistoryList');
    if (!template || !list) {
        displayNotification('Can-t find html code', { type: "error" })
        return;
    }

    const clone = template.content.cloneNode(true) as HTMLElement;
    (clone.querySelector('#matchGame') as HTMLElement).textContent = game;
    (clone.querySelector('#matchOpponent') as HTMLElement).textContent = opponent;
    (clone.querySelector('#opponentAvatar') as HTMLElement).textContent = opponentAvatar;
    (clone.querySelector('#matchScore1') as HTMLElement).textContent = `${score1}`;
    (clone.querySelector('#matchScore2') as HTMLElement).textContent = `${score2}`;
    (clone.querySelector('#matchResult') as HTMLElement).textContent = result;
    (clone.querySelector('#matchTime') as HTMLElement).textContent = matchTime;

    if (result === 'DEFEAT') {
        (clone.querySelector('#matchResult') as HTMLElement).classList.add("text-red-700");
        } else {
        (clone.querySelector('#matchResult') as HTMLElement).classList.add("text-green-700");
    }
    const item = clone.querySelector('li');
    if (item && result === 'DEFEAT') {
        item.classList.remove("to-green-800");
        item.classList.add("to-red-800");
    }
    if (item && game === 'Tower-Defense') {
        item.classList.remove("from-yellow-700");
        item.classList.add("from-blue-700");
    }
    list.appendChild(clone);
}

async function displayCombinedMatchHistory(matches: { match: MatchResult, game: string }[], id: number, data: ChartData[]) {
    for (const { match, game } of matches) {
        let opponent: UserData[] = [];
        const oppId = match.playerA_id === id ? match.playerB_id : match.playerA_id;

        if (oppId !== -1) {
            opponent = await fetchUserInformation([oppId]);
        } else {
            opponent = [{
                id: -1,
                online: true,
                nickName: 'Guest',
                avatar: '../images/login.png'
            }];
        }
        const result: string = (id === match.winner_id) ? 'VICTORY' : 'DEFEAT';
        const scoreUser: number[] = match.playerA_id === id ? [match.scoreA, match.scoreB] : [match.scoreB, match.scoreA];
        const idGame: number = 'match-server' == game ? 0 : 1;
        if (id === match.winner_id) {
            data[idGame].wins += 1;
            data[idGame].mmr += 27;
        } else {
            data[idGame].losses += 1;
            data[idGame].mmr -= 15;
        }
        addMatchEntry(game, opponent[0].nickName, opponent[0].avatar, scoreUser[0], scoreUser[1], result, match.match_time);
    }
}

export async function getGameHistory (game: string): Promise<MatchResult[] | undefined> {
    try {
        const token = sessionStorage.getItem('token')
        const response = await fetch(`/${game}/getMatchHistory`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })
        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            displayNotification(`${game} History not loaded`, { type: "error" })
            return undefined
        }
        return await response.json()
    } catch (error) {
        console.error(error)
    }
}

export async function printMatchHistory() {
    // const id =
    const id = sessionStorage.getItem('id');
    if (!id) {
        await navigate('/home')
        displayNotification('Could not find your match result', { type: "error" })
        return;
    }
    const pongMH: MatchResult[] | undefined = await getGameHistory('match-server');
    const tdMH: MatchResult[] | undefined = await getGameHistory('tower-defense');
    const idNum = Number(id)
    const combined: { match: MatchResult, game: string }[] = [];

    (pongMH || []).forEach(match => combined.push({ match, game: 'Pong' }));
    (tdMH || []).forEach(match => combined.push({ match, game: 'Tower-Defense' }));
    combined.sort((a, b) => new Date(b.match.match_time).getTime() - new Date(a.match.match_time).getTime());

    const data: ChartData[] = [
        {
            wins: 0,
            losses: 0,
            mmr: 0,
            canvasId: "donutChartPong",
            mmrElementId: "mmrPong"
        },
        {
            wins: 0,
            losses: 0,
            mmr: 0,
            canvasId: "donutChartTd",
            mmrElementId: "mmrTd"
        }
    ];
    await displayCombinedMatchHistory(combined, idNum, data);
    drawChart(data)
}


function drawChart(data: ChartData[]) {
    const winRateTd = document.getElementById('winRateTd')
    if (winRateTd && data[0].wins + data[0].losses == 0) winRateTd.innerText = 'N/A'
    else if (winRateTd) winRateTd.innerText = 'Win Rate - ' + ((data[0].wins / (data[0].wins + data[0].losses)) * 100).toFixed(2) + '%';
    const winRatePong = document.getElementById('winRatePong')
    if (winRatePong && data[0].wins + data[0].losses == 0) winRatePong.innerText = 'N/A'
    else if (winRatePong) winRatePong.innerText = 'Win Rate - ' + ((data[0].wins / (data[0].wins + data[0].losses)) * 100).toFixed(2) + '%';
}