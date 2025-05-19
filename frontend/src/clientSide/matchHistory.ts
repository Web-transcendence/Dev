import {displayNotification} from "./notificationHandler.js";
import {fetchUserInformation, UserData} from "./user.js";
import {navigate} from "./front.js";

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

async function getElementsOfMatch(MatchResult: MatchResult[] | undefined, id: number, game: string) {
    try {
        if (MatchResult === undefined) {
            console.log('No match found.', game);
            displayNotification(`You Didn't played a single game of ${game}`, { type: "error" })
            return;
        }
        for (const match of MatchResult) {
            let opponent: UserData[] = [];
            console.log('opponent:', opponent);
            const oppId=  match.playerA_id === id ? match.playerB_id : match.playerA_id;
            console.log('matchtd', match.playerA_id, match.playerB_id);
            if (oppId !== -1)
                opponent = await fetchUserInformation([oppId]);
            else
                opponent = [{
                    id: -1,
                    online: true,
                    nickName: 'Guest',
                    avatar: '../images/login.png'
                }];
            let result: string = 'DEFEAT';
            if (id == match.winner_id)
                result = 'VICTORY';
            addMatchEntry(game, opponent[0].nickName, opponent[0].avatar, match.scoreA, match.scoreB, result, match.match_time)
        }
    } catch (e) {
        console.error(e)
        displayNotification('Could not find match result', { type: "error" })
        return ;
    }
}

export async function getGameHistory (id: string, game: string): Promise<MatchResult[] | undefined> {
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
    const id = sessionStorage.getItem('id');
    if (!id) {
        await navigate('/home')
        displayNotification('Could not find your match result', { type: "error" })
        return;
    }
    const pongMH: MatchResult[] | undefined = await getGameHistory(id, 'match-server');
    const tdMH: MatchResult[] | undefined = await getGameHistory(id, 'tower-defense');
    const idNum = Number(id)
    await getElementsOfMatch(pongMH, idNum, 'Pong')
    await getElementsOfMatch(tdMH, idNum, 'Tower-Defense')
}

