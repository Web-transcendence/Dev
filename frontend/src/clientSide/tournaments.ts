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
    console.log(`joinTournaments: ${nameTournament}`);
    const name = document.getElementById('nameTournaments');
    if (name) name.innerText = nameTournament;
    const tournamentList: {participants: number[], maxPlayer: number, status: string}[] | undefined = await getTournamentList()
    if (!tournamentList) {
        DispayNotification(`Error Can't find Tournaments`);
        await navigate('/home')
        return ;
    }
    console.log('GTL', tournamentList)
    let player = 0;
    const playerList = document.getElementById('playerList')
    const playerTmp = document.getElementById('playerTemplate')  as HTMLTemplateElement | null;
    const section = tournamentList.find(s => s.maxPlayer === nbrTournament);
    if ( section && playerList && playerTmp ) {
        for (const participants of section.participants) {
            const [userData]: UserData[] = await fetchUserInformation(section.participants);
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
            const span = item.querySelector('span');
            if (span) span.id = `friendId-${participants}`;
            const img = item.querySelector('img');
            if (img) img.id = `friendId-${userData.id}`;
            playerList.appendChild(item);
            player++;
        }
    }
    console.log(`joinTournaments: ${player}`);
    for (; player < nbrTournament; player++) {
        console.log('ndrPLAYER',player);
        const emptySlotTmp = document.getElementById('emptySlotTemplate')  as HTMLTemplateElement | null;
        if (!emptySlotTmp ) {
            DispayNotification('Error 3 occur, please refresh your page.');
            return ;
        }
        if (!playerTmp) {
            DispayNotification('Error 4 occur, please refresh your page.');
            return ;
        }
        if (!playerList) {
            DispayNotification('Error 5 occur, please refresh your page.');
            return ;
        }
        const clone = emptySlotTmp.content.cloneNode(true);
        if (!clone) {
            DispayNotification('Error 6 occur, please refresh your page.');
            return ;
        }
        playerList.appendChild(clone);
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