import { navigate } from './front.js'
import { fetchUserInformation, getTournamentList, UserData } from './user.js'
import { displayNotification } from './notificationHandler.js'

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
	const name = document.getElementById('nameTournament')
	if (name) name.innerText = nameTournament
	const tournamentList: {
		participants: number[],
		maxPlayer: number,
		status: string
	}[] | undefined = await getTournamentList()
	if (!tournamentList) {
		displayNotification(`Error Can't find Tournaments`)
		await navigate('/home')
		return
	}
	let player = 0
	const myId = Number(sessionStorage.getItem('id'))
	const playerList = document.getElementById('playerList')
	const playerTmp = document.getElementById('playerTemplate') as HTMLTemplateElement | null
	const section = tournamentList.find(s => s.maxPlayer === nbrTournament)
	if (section && playerList && playerTmp) {
		const userData: UserData[] = await fetchUserInformation(section.participants)
		for (const { id, nickName, avatar } of userData) {
			const clone = playerTmp.content.cloneNode(true) as HTMLElement | null
			if (!clone) {
				displayNotification('Error 1 occur, please refresh your page.')
				return
			}
			const item = clone.querySelector('li')
			if (!item) {
				displayNotification('Error 2 occur, please refresh your page.')
				return
			}
			item.id = `itemId-${id}`
			const span = item.querySelector('span')
			if (span) {
				span.id = `spanId-${id}`
				console.log('logname', nickName)
				span.innerText = nickName
			}
			const img = item.querySelector('img')
			if (img) {
				img.id = `imgId-${id}`
				if (avatar) img.src = avatar
				else img.src = '../images/login.png'
			}
			playerList.appendChild(item)
			player++
		}
	}
	const numberOfPlayer = document.getElementById(`numberOfPlayer`)
	if (numberOfPlayer) {
		const players = document.querySelectorAll('#playerList li')
		const number = players.length
		numberOfPlayer.innerText = `${number}/${nbrTournament}`
	}
}

export async function quitTournaments() {
	const token = sessionStorage.getItem('token')

	const response = await fetch(`/tournament/quit`, {
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

export async function getBrackets(nbrPlayer: number) {
	const brackets = document.getElementById('brackets')
	for (let nbr: number = nbrPlayer; nbr >= 2;) {
		const playerListTmp = document.getElementById('playerListTemplate')
		if (!brackets) {
			displayNotification(`Error from html1`)
			await navigate('/home')
			return
		}
		if (!playerListTmp) {
			displayNotification(`Error from html2`)
			await navigate('/home')
			return
		}
		const clone = playerListTmp.cloneNode(true) as HTMLElement | null
		if (!clone) {
			displayNotification(`Error from html3`)
			await navigate('/home')
			return
		}
		clone.id = `playerList-${nbr}`
		brackets.appendChild(clone)
		nbr = nbr / 2
	}
	const players = [
		'Alice', 'Bob',
		'Charlie', 'David',
	]
	for (const player of players) {
		const playerList = document.getElementById(`playerList-${nbrPlayer}`)
		const playerTmp = document.getElementById('playerTemplate')
		if (!playerList || !playerTmp) {
			displayNotification(`Error from html4`)
			await navigate('/home')
			return
		}
		const clone = playerList.cloneNode(true) as HTMLElement | null
		if (!clone) {
			displayNotification(`Error from html5`)
			await navigate('/home')
			return
		}
		const item = clone.querySelector('#player')
		if (!item) {
			displayNotification(`Error from html6`)
			await navigate('/home')
			return
		}
		item.innerHTML = player
		item.id = player
		playerList.appendChild(item)
	}

}

export async function fetchTournamentBrackets(tournamentId: number): Promise<{id1: number, id2:number }[] | undefined > {
	const token = sessionStorage.getItem('token')
    console.log('TOUR nbr id:', tournamentId)
	const response = await fetch(`/tournament/logTournamentStep/${tournamentId}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'authorization': 'Bearer ' + token
		}
	})
	if (!response.ok) {
		const error = await response.json()
		console.error(error.error)
		displayNotification(error.error)
        return undefined
	}
    console.log('Success fetchTournamentBrackets')
    const ret = await response.json()
    console.log(ret)
    return ret
	// return response.json()
}
