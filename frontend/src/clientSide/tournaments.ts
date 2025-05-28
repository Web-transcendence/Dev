import { navigate } from './front.js'
import { fetchUserInformation, getTournamentList, UserData } from './user.js'
import { displayNotification } from './notificationHandler.js'
import { path } from './front.js'

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
			displayNotification(error.error, { type: 'error' })
		} else displayNotification('You have joined a tournament!')
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
	if (!tournamentList) return
	console.log('TournamentList', tournamentList)
	let player = 0
	const playerList = document.getElementById('playerList')
	if (!playerList) return
	playerList.innerHTML = ''
	console.log('Cleared playerList, current length:', playerList.querySelectorAll('li').length)

	const playerTmp = document.getElementById('playerTemplate') as HTMLTemplateElement | null
	const section = tournamentList.find(s => s.maxPlayer === nbrTournament)
	if (section && playerList && playerTmp) {
		console.log('Number of participants:', section?.participants.length)
		console.log('Player list children before appending:', playerList.children.length)
		if (section.status === 'started') {
			console.log('TOUR HAVE STARTED')
			if (path != '/brackets') {
				console.log('redirect to brackets')
				await navigate('/brackets')
			}
			console.log('no redirection to brackets')
			return
		}
		const userData: UserData[] = await fetchUserInformation(section.participants)
		console.log('TournamentuserData', userData)
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

export async function quitTournaments(idTournament: number) {
	const token = sessionStorage.getItem('token')
	console.log(idTournament)
	const response = await fetch(`/tournament/quit/${idTournament}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'authorization': 'Bearer ' + token
		}
	})
	if (!response.ok) {
		const error = await response.json()
		console.error(error.error)
		displayNotification(error.error, { type: 'error' })
	} else displayNotification('You have left a tournament!')
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

export async function fetchTournamentBrackets(tournamentId: number): Promise<{
	id1: number,
	id2: number
}[] | undefined> {
	const token = sessionStorage.getItem('token')
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
	const ret = await response.json()
	console.log(ret)
	return ret
}
