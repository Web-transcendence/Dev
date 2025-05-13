import { navigate } from './front.js'
import {fetchUserInformation, getTournamentList, UserData} from "./user.js";
import {DispayNotification} from "./notificationHandler.js";

export async function joinTournament(tournamentId: number) {
    try {
        const token = sessionStorage.getItem('token')
        const response = await fetch(`/tournament/join`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({ tournamentId })
        })

        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
        }
    } catch (error) {
        console.error(error)
    }
}

export async function displayTournaments(nbrTournament: number, nameTournament: string) {
    const name = document.getElementById('nameTournament');
    if (name) name.innerText = nameTournament;
    const tournamentList: {participants: number[], maxPlayer: number, status: string}[] | undefined = await getTournamentList()
    if (!tournamentList) {
        DispayNotification(`Error Can't find Tournaments`);
        await navigate('/home')
        return ;
    }
    let player = 0;
    const myId = Number(sessionStorage.getItem('id'))
    const playerList = document.getElementById('playerList')
    const playerTmp = document.getElementById('playerTemplate')  as HTMLTemplateElement | null;
    const section = tournamentList.find(s => s.maxPlayer === nbrTournament);
    if ( section && playerList && playerTmp ) {
        const userData: UserData[] = await fetchUserInformation(section.participants);
        for (const {id, nickName, avatar} of userData) {
            const clone = playerTmp.content.cloneNode(true) as HTMLElement | null;
            if (!clone) {
                DispayNotification('Error 1 occur, please refresh your page.');
                return;
            }
            const item = clone.querySelector("li");
            if (!item) {
                DispayNotification('Error 2 occur, please refresh your page.');
                return;
            }
            item.id = `itemId-${id}`
            const span = item.querySelector('span');
            if (span) {
                span.id = `spanId-${id}`;
                console.log('logname', nickName)
                span.innerText = nickName;
            }
            const img = item.querySelector('img');
            if (img) {
                img.id = `imgId-${id}`;
                if (avatar) img.src = avatar;
                else img.src = '../images/login.png';
            }
            playerList.appendChild(item);
            player++;
        }
    }
    const numberOfPlayer = document.getElementById(`numberOfPlayer`);
    if (numberOfPlayer) {
        const players = document.querySelectorAll("#playerList li");
        const number = players.length;
        numberOfPlayer.innerText = `${number}/${nbrTournament}`;
    }
    document.getElementById('launchTournamentBtn')?.addEventListener('click', event => navigate('/launchTournaments', event));
}

export async function quitTournaments() {
    const token = sessionStorage.getItem('token')

    const response = await  fetch(`/tournament/quit`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': 'Bearer ' + token
        }
    })
    if (!response.ok) {
        const error = await response.json()
        console.error(error.error)
    }
}

export async function launchTournament() {
    try {
        const token = sessionStorage.getItem('token')
        const response = await fetch(`/tournament/launch`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })

        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            console.log(error)
        }
    } catch (error) {
        console.error(error)
    }
}