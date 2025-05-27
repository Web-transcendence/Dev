import {displayNotification} from "./notificationHandler.js";
import {fetchUserInformation, UserData} from "./user.js";
import {navigate} from "./front.js";

interface ChartData {
    wins: number;
    losses: number;
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
    (clone.querySelector('#opponentAvatar') as HTMLImageElement).src = opponentAvatar;
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

        if (oppId !== -1 && oppId !== -2 ) {
            opponent = await fetchUserInformation([oppId]);
        }
        else if (oppId == -1) {
            opponent = [{
                id: -1,
                online: true,
                nickName: 'Guest',
                avatar: '../images/logout.png'
            }];
            }
        else {
                opponent = [{
                    id: -2,
                    online: true,
                    nickName: 'AI',
                    avatar: '../images/AI.png'
                }];
            }
        if (game === 'Tower-Defense') {
            if (id === match.winner_id) data[1].wins++
            else data[1].losses++
        }
        else {
            if (id === match.winner_id) data[0].wins++
            else data[0].losses++
        }


        const result: string = (id === match.winner_id) ? 'VICTORY' : 'DEFEAT';
        const scoreUser: number[] = match.playerA_id === id ? [match.scoreA, match.scoreB] : [match.scoreB, match.scoreA];
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
    const id = await getId()
    console.log('id found is ', id)
    if (!id) return;
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
        },
        {
            wins: 0,
            losses: 0,
        }
    ];
    await displayCombinedMatchHistory(combined, idNum, data)
    await drawChart(data)
}


async function drawChart(data: ChartData[]) {
    const winRatePong = document.getElementById('winRatePong')
    if (winRatePong && data[0].wins + data[0].losses != 0) winRatePong.innerText = 'Win Rate - ' + ((data[0].wins / (data[0].wins + data[0].losses)) * 100).toFixed(2) + '%'
    const winRateTd = document.getElementById('winRateTd')
    if (winRateTd && data[1].wins + data[1].losses != 0) winRateTd.innerText = 'Win Rate - ' + ((data[1].wins / (data[1].wins + data[1].losses)) * 100).toFixed(2) + '%'

    const mmrPong = document.getElementById('mmrPong')
    const mmrTd = document.getElementById('mmrTd')
    const log = await getMmrById()
    if (!log) {
        displayNotification(`Error can't find your mmr`)
        await navigate('/home')
        return
    }
    if (mmrPong) mmrPong.innerText = 'MMR - ' + log.pongMmr
    if (mmrTd) mmrTd.innerText = 'MMR - ' + log.tdMmr

}

const getMmrById = async (): Promise<{ pongMmr: number, tdMmr: number } | undefined> => {
    const token = sessionStorage.getItem('token')
    try {

    const response = await fetch(`/user-management/mmr`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': 'Bearer ' + token,
        },
    });
    if(!response.ok) {
        throw new Error(`this nickname doesn't exist`);
    }
    const {pongMmr, tdMmr} = await response.json() as {pongMmr: number, tdMmr: number};
    return {pongMmr, tdMmr};
    }
    catch (error) {
        console.log(error)
        return undefined
    }
}

export async function getId(): Promise<string | undefined> {
    const token = sessionStorage.getItem('token')
    const response = await fetch(`/authJWT`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': 'Bearer ' + token,
        },
    })
    if (!response.ok) {
        await navigate('/home')
        displayNotification('Error no authorization', {type: "error"})
        return undefined;
    }
    const {id} = await response.json()
    return id
}